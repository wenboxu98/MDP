import torch
import torch.nn as nn
from torch.nn import Parameter
import random
import numpy as np
class CrossVideoLoss(nn.Module):
    def __init__(self):
        super(CrossVideoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )
        
        loss = HA_refinement + HB_refinement
        return loss   

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.cv_criterion = CrossVideoLoss()

    def forward(self, video_scores, label, contrast_pairs, pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_cv = self.cv_criterion(contrast_pairs)
        
        # cross video loss
        for i,vs in enumerate(pairs):
            temp_pairs = {
            'EA': torch.stack([contrast_pairs['EA'][list(pairs[i])[0]],contrast_pairs['EA'][list(pairs[i])[1]]]),
            'HA': torch.stack([contrast_pairs['HA'][list(pairs[i])[1]],contrast_pairs['HA'][list(pairs[i])[0]]]),
            'EB': torch.stack([contrast_pairs['EB'][list(pairs[i])[0]],contrast_pairs['EB'][list(pairs[i])[1]]]),
            'HB': torch.stack([contrast_pairs['HB'][list(pairs[i])[1]],contrast_pairs['HB'][list(pairs[i])[0]]]),
            }
            loss_cv += 0.2/len(pairs) * self.cv_criterion(temp_pairs)
        ## modified by xuwb 20241107 start
        ## remove loss_cv

        # loss_total = loss_cls + 0.01 * loss_cv
        loss_total = loss_cls
        ## modified by xuwb 20241107 end
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/cv': loss_cv,
            'Loss/Total': loss_total
        }
        
        return loss_total, loss_dict

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()
    def process_kl_distance(self, kl_distance, utt_scores):
        batch_size = utt_scores.shape[0]
        result = torch.zeros((batch_size, 2)).cuda()
        for i in range(batch_size):
            if utt_scores[i][0] > utt_scores[i][1]:
                result[i][0] = kl_distance[i]
            else:
                result[i][1] = kl_distance[i]
        return result

    def forward(self, kl_distance, label, utt_scores):
        label = label / torch.sum(label, dim=1, keepdim=True)
        kl_distance = self.process_kl_distance(kl_distance, utt_scores)
        loss = self.bce_criterion(kl_distance, label)
        return loss
class DPLoss(nn.Module):
    def __init__(self):
        super(DPLoss, self).__init__()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        # self.bce_criterion = nn.BCELoss()
    def forward(self, input_score, target):
        if (target.shape[1]!=1): ## one-hot (batchsize, num_class)
          target = torch.argmax(target, dim=1, keepdim=True)
        target = target.float() #.view(-1, 1)
        input_score = input_score.unsqueeze(-1)
        loss = self.bce_criterion(input_score, target)      
        return loss
class TotalLoss_weak(nn.Module):
    def __init__(self):
        super(TotalLoss_weak, self).__init__()
        self.utt_criterion = P2SGradLoss() # p2sgrad.P2SGradLoss() #ActionLoss() #
        self.kl_criterion = KLLoss()
        self.dp_criterion = DPLoss()

    def forward(self, utt_scores, utt_labels, dp_distance):
        # added by xuwb 20241222 start

        # added by xuwb 20241222 end
        #############
        loss_utt_cls = self.utt_criterion(utt_scores, utt_labels)
        

        # added by xuwb 20250408 start
        # print(dp_distance)
        dp_loss = self.dp_criterion(dp_distance, utt_labels)
        dp_loss_weight = 0.2
        # added by xuwb 20250408 end
        # modified by xuwb 20250115 start 
        """for ablation study"""
        # kl_loss = self.kl_criterion(kl_distance, utt_labels)
        # kl_loss_weight = 0.01

        #############
        loss_total = loss_utt_cls + dp_loss_weight * dp_loss
        
        # loss_total = loss_utt_cls
        # modified by xuwb 20250115 end
        loss_dict = {
            'Loss/utt': loss_utt_cls,
            'Loss/dp': dp_loss,
            'Loss/Total': loss_total
        }
        
        return loss_total, loss_dict
    
class P2SActivationLayer(nn.Module):
    """ Output layer that produces cos\theta between activation vector x
    and class vector w_j
    slack: P2SActivationLayer is just the name in the code, but not a 
    layer type like LSTM or pooling.
    It is a fully-connected layer 

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    
    Method: cos\theta = forward(x)
    
    x: (batchsize, input_dim)
    
    cos: (batchsize, output_dim)
          where \theta is the angle between
          input feature vector x[i, :] and weight vector w[j, :]
    """
    def __init__(self, in_planes, out_planes):
        super(P2SActivationLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)

        output:
        -------
          tensor (batchsize, output_dim)
          # for cosine similarity, higher is more similar
          # and score[spoof, bonafide], we use score[:,1] (bonafide score) as final results.
          
        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5).to(input_feat.device)
        
        # normalize the input feature vector
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # done
        return cos_theta

    
class P2SGradLoss(nn.Module):
    """
    P2SGradLoss()
    
    Just MSE loss between output and target one-hot vectors
    """
    def __init__(self, weight=False, label_reverse=False, reverse_ratio = 0.1):
        super(P2SGradLoss, self).__init__()
        self.m_loss = nn.MSELoss()
        self.label_reverse = label_reverse
        self.reverse_ratio = reverse_ratio

    def forward(self, input_score, target):
        """ 
        input
        -----
          input_score: tensor (batchsize, class_num)
                 cos\theta
        output
        ------
          loss: scaler
        """
        # target (batchsize, 1)
        
        if (target.shape[1]!=1): ## one-hot (batchsize, num_class)
          target = torch.argmax(target, dim=1, keepdim=True)
        target = target.long() #.view(-1, 1)
        
        # filling in the target
        # index (batchsize, class_num)
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            # index[i][target[i][j]] = 1
            #print(index.shape, target.data.shape)
            index.scatter_(1, target.data.view(-1, 1), 1)

            if(self.label_reverse):
                random.seed(index[:,1].sum().tolist())  # rebuild.
                reverse_select_idx = random.sample(np.arange(index.shape[0]).tolist(),int(index.shape[0]*self.reverse_ratio))
                index[reverse_select_idx] = 1-index[reverse_select_idx]
    
        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(input_score, index)

        return loss