import torch
import torch.nn as nn
class ForgeryActivationModule(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(ForgeryActivationModule, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=1024, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        # added by xuwb 20250408 start
        feature_former = embeddings[:, :-1, :]
        feature_later = embeddings[:, 1:, :]
        squared_diffs = (feature_former - feature_later) ** 2
        mse_per_pair = torch.mean(squared_diffs, dim=2)
        # added by xuwb 20250408 end
        out = self.dropout(out)
        out = self.f_cls(out)
        cas = out.permute(0, 2, 1)
        aness = cas.sum(dim=2)
        return embeddings, cas, aness, mse_per_pair
class visual_Module(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(visual_Module, self).__init__()
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        return embeddings
    
class audio_Module(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(audio_Module, self).__init__()
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        return embeddings

class CrossModalAttention(nn.Module):
    def __init__(self, input_dim_audio, input_dim_visual, attention_dim):
        super(CrossModalAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim_visual, attention_dim)  # W_q
        self.key_layer = nn.Linear(input_dim_audio, attention_dim)  # W_k
        self.value_layer = nn.Linear(input_dim_audio, attention_dim)  # W_v
        self.output_layer = nn.Linear(attention_dim, input_dim_visual)  

    def forward(self, visual, audio):
        # get the shape of visual and audio
        batch_size, channels, feature_visual = visual.shape
        batch_size, channels, feature_audio = audio.shape

        # Step 1: flat the feature
        visual_flat = visual.view(batch_size, channels, feature_visual)  # [batch_size, channels, feature_video]
        audio_flat = audio.view(batch_size, channels, feature_audio)  # [batch_size, channels, feature_audio]

        # Step 2: calculate Q, K, V
        query = self.query_layer(visual_flat)  # [batch_size, channels, attention_dim]
        key = self.key_layer(audio_flat)  # [batch_size, channels, attention_dim]
        value = self.value_layer(audio_flat)  # [batch_size, channels, attention_dim]

        # (Q * K^T) / (d^(1/2))
        attention_scores = torch.matmul(query, key.transpose(1, 2))  # [batch_size, channels, channels]
        attention_scores = attention_scores / (feature_audio ** 0.5)  

        # Step 3: S(.)
        attention_scores_sum = attention_scores.sum(dim=1, keepdim=True)  # [batch_size, 1, channels]

        # Step 4: change the shape of attention_scores_sum into [batch_size, channels, 1] 
        attention_scores_sum = attention_scores_sum.transpose(2, 1)  # [batch_size, channels, 1]

        # Step 5: calculate ATT_v

        weighted_value = attention_scores_sum * value  # [batch_size, channels, attention_dim]

        # Step 6: obtained the output
        attention_output = self.output_layer(weighted_value)  # [batch_size, channels, feature_video]

        return attention_output

       
# MDP Pipeline
class MDP(nn.Module):
    def __init__(self, cfg):
        super(MDP, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.aness_module = ForgeryActivationModule(cfg.FEATS_DIM, cfg.NUM_CLASSES)
        '''
        for multimodal fusion
        ''' 
        self.softmax = nn.Softmax(dim=1)

        self.r_easy = cfg.R_EASY
        self.r_hard = cfg.R_HARD
        self.m = cfg.m
        self.M = cfg.M

        self.dropout = nn.Dropout(p=0.6)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1024)

        self.visual_embed = visual_Module(1024, 512)
        self.visual_layerNorm = nn.LayerNorm(512)
        self.audio_embed = audio_Module(1024, 512)
        self.audio_layerNorm = nn.LayerNorm(512)
        self.cross_attention_audio2visual = CrossModalAttention(input_dim_audio=512, input_dim_visual=512, attention_dim=1024)
        self.cross_attention_audio2visual_layerNorm = nn.LayerNorm(512)
        self.cross_attention_visual2audio = CrossModalAttention(input_dim_audio=512, input_dim_visual=512, attention_dim=1024)
        self.cross_attention_visual2audio_layerNorm = nn.LayerNorm(512)
    
    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores
    # added by xuwb 20250408 start
    def get_video_cls_scores_mse(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy]
        mse_scores = topk_scores.mean(1)
        # video_scores = self.softmax(topk_scores.mean(1))
        return mse_scores    
    # added by xuwb 20250408 end



    def forward(self, visual_data, audio_data):
        visual_data = self.adaptive_pool(visual_data)

        visual_data = self.visual_embed(visual_data)
        visual_data = self.visual_layerNorm(visual_data)

        audio_data = self.audio_embed(audio_data)
        audio_data = self.audio_layerNorm(audio_data)

        attention_visual = self.cross_attention_audio2visual(visual_data, audio_data)
        attention_visual = self.cross_attention_audio2visual_layerNorm(attention_visual)
        attention_audio = self.cross_attention_visual2audio(audio_data, visual_data)
        attention_audio = self.cross_attention_visual2audio_layerNorm(attention_audio)
        x = torch.cat((audio_data, visual_data, attention_visual, attention_audio), dim=2)
        num_segments = x.shape[1]

        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard

        embeddings, cas, aness, mse_per_pair = self.aness_module(x)
        
        video_scores = self.get_video_cls_scores(cas, k_easy)
        mse_scores = self.get_video_cls_scores_mse(mse_per_pair, k_easy)
        
        if self.training:
            return video_scores, aness, cas, mse_scores
        
        return video_scores, aness, cas
