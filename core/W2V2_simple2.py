import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from .FDN import *
# from .registry import register_model

# MODEL_REGISTRY = {}

# def register_model(name):
#     def wrapper(cls):
#         MODEL_REGISTRY[name] = cls
#         return cls
#     return wrapper

# def get_model(name):
#     if name in MODEL_REGISTRY:
#         return MODEL_REGISTRY[name]()
#     else:
#         raise ValueError(f"Model {name} not found!")



# @register_model("W2V2_simple_pre")
class Simple_Module(nn.Module):
    def __init__(self, len_feature=1024, num_classes=2):
        super(Simple_Module, self).__init__()
        self.len_feature = len_feature

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        # self.asr=ASRModel()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.LL=torch.nn.Linear(1024,128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.filts = [128, [1, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128]]
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=self.filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=self.filts[2])),
            nn.Sequential(Residual_block(nb_filts=self.filts[3])),
            nn.Sequential(Residual_block(nb_filts=self.filts[4])),
            nn.Sequential(Residual_block(nb_filts=self.filts[5])),
            nn.Sequential(Residual_block(nb_filts=self.filts[6])))
        self.selu = nn.SELU(inplace=True)
        self.att_dim=[self.filts[-1][-1],64]
        self.attention = nn.Sequential(
            nn.Conv2d(self.att_dim[0], self.att_dim[1], kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(self.att_dim[1]),
            nn.Conv2d(self.att_dim[1], self.att_dim[0], kernel_size=(1,1)),
        )
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.7)
        self.softmax = nn.Softmax(dim=1)


    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        # print('------------')
        # print(cas.shape, cas)
        # print(sorted_scores.shape, sorted_scores)
        topk_scores = sorted_scores[:, :k_easy, :]
        utt_scores = self.softmax(topk_scores.mean(1))
        # print(topk_scores.mean(1), utt_scores)
        return utt_scores

    def forward(self, x):
        # x = self.asr(x)
        x = self.LL(x)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x) 
        x = self.norm(x)
        x = self.selu(x)
        w = self.attention(x) 
        w_C = F.softmax(w, dim=-1)
        embeddings = torch.sum(x * w_C, dim=-1).transpose(1,2) #B,T,128
        # print(embeddings.shape)
        out = self.f_cls(embeddings.permute(0, 2, 1)) #[B,128,T] [B,2,T] [B,T,2]
        cas = out.permute(0, 2, 1)
        utt_score=self.get_video_cls_scores(cas, k_easy=50)
        return utt_score, cas 


# @register_model("W2V2_simple2_pre")
class Simple_Module2(nn.Module):
    def __init__(self, len_feature=1024, num_classes=2):
        super(Simple_Module2, self).__init__()
        self.len_feature = len_feature

        # self.f_cls = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.ReLU()
        # )

        self.f_cls = torch.nn.Linear(128,2)
        
        # self.asr=ASRModel()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.LL=torch.nn.Linear(1024,128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.filts = [128, [1, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128]]
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=self.filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=self.filts[2])),
            nn.Sequential(Residual_block(nb_filts=self.filts[3])),
            nn.Sequential(Residual_block(nb_filts=self.filts[4])),
            nn.Sequential(Residual_block(nb_filts=self.filts[5])),
            nn.Sequential(Residual_block(nb_filts=self.filts[6])))
        self.selu = nn.SELU(inplace=True)
        self.att_dim=[self.filts[-1][-1],64]
        self.attention = nn.Sequential(
            nn.Conv2d(self.att_dim[0], self.att_dim[1], kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(self.att_dim[1]),
            nn.Conv2d(self.att_dim[1], self.att_dim[0], kernel_size=(1,1)),
        )
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.7)
        self.softmax = nn.Softmax(dim=1)


    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        # print('------------')
        # print(cas.shape, cas)
        # print(sorted_scores.shape, sorted_scores)
        topk_scores = sorted_scores[:, :k_easy, :]
        utt_scores = self.softmax(topk_scores.mean(1))
        # print(topk_scores.mean(1), utt_scores)
        return utt_scores

    def forward(self, x):
        # x = self.asr(x)
        x = self.LL(x)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x) 
        x = self.norm(x)
        x = self.selu(x)
        w = self.attention(x) 
        w_C = F.softmax(w, dim=-1)
        embeddings = torch.sum(x * w_C, dim=-1).transpose(1,2) #B,T,128
        cas = self.f_cls(embeddings) #[B,128,T] [B,2,T] [B,T,2]
        # print(embeddings.shape)
        utt_score=self.get_video_cls_scores(cas, k_easy=50)
        return utt_score, cas 





# @register_model("W2V2_simple3_pre")
class Simple_Module3(nn.Module):
    def __init__(self, len_feature=1024, num_classes=2):
        super(Simple_Module3, self).__init__()
        self.len_feature = len_feature

        # self.f_cls = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.ReLU()
        # )

        self.f_cls = aMLP(d_model=128, d_ffn=-2, seq_len=1070, gmlp_layers = 1, batch_first=True)
        self.f_cls2 =nn.Linear(128,2)
        # self.asr=ASRModel()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.LL=torch.nn.Linear(1024,128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.filts = [128, [1, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128]]
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=self.filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=self.filts[2])),
            nn.Sequential(Residual_block(nb_filts=self.filts[3])),
            nn.Sequential(Residual_block(nb_filts=self.filts[4])),
            nn.Sequential(Residual_block(nb_filts=self.filts[5])),
            nn.Sequential(Residual_block(nb_filts=self.filts[6])))
        self.selu = nn.SELU(inplace=True)
        self.att_dim=[self.filts[-1][-1],64]
        self.attention = nn.Sequential(
            nn.Conv2d(self.att_dim[0], self.att_dim[1], kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(self.att_dim[1]),
            nn.Conv2d(self.att_dim[1], self.att_dim[0], kernel_size=(1,1)),
        )
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.7)
        self.softmax = nn.Softmax(dim=1)


    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        # print('------------')
        # print(cas.shape, cas)
        # print(sorted_scores.shape, sorted_scores)
        topk_scores = sorted_scores[:, :k_easy, :]
        utt_scores = self.softmax(topk_scores.mean(1))
        # print(topk_scores.mean(1), utt_scores)
        return utt_scores

    def forward(self, x):
        # x = self.asr(x)
        x = self.LL(x)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x) 
        x = self.norm(x)
        x = self.selu(x)
        w = self.attention(x) 
        w_C = F.softmax(w, dim=-1)
        embeddings = torch.sum(x * w_C, dim=-1).transpose(1,2) #B,T,128
        cas = self.f_cls(embeddings) #[B,128,T] [B,2,T] [B,T,2]
        cas = self.f_cls2(cas) #[B,128,T] [B,2,T] [B,T,2]
        # print(embeddings.shape)
        utt_score=self.get_video_cls_scores(cas, k_easy=50)
        return utt_score, cas 