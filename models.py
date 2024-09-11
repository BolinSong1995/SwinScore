import torch
from torch import nn
import math
from typing import Union, Sequence, Tuple
from utils import define_act_layer
from swintransformer import SwinTransformer, look_up_option
from einops import rearrange
import os 
from medpy.io import load, header
import numpy as np



class MultiTaskModel(nn.Module):
    def __init__(self, task, in_features, hidden_units=None, act_layer=nn.ReLU(), dropout=0.7) -> None:

        super().__init__()
        self.act = act_layer
        incoming_features = in_features
        hidden_layer_list = []
        self.task = task
        for hidden_unit in hidden_units:
            hidden_block = nn.Sequential(
                nn.Linear(incoming_features, hidden_unit),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(hidden_unit),
                nn.Dropout(dropout),
            )
            hidden_layer_list.append(hidden_block)
            incoming_features = hidden_unit
        self.hidden_layer = nn.Sequential(*hidden_layer_list)
        out_features = 2 if self.task=="multitask" else 1
        self.classifier = nn.Linear(hidden_units[-1], out_features)
        # self.output_act = nn.Sigmoid()
        # self.output_act1 = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.hidden_layer(x)
        classifier = self.classifier(x)
        # print(classifier)
        if self.task =="multitask":
            grade, hazard = classifier[0], classifier[1]
            return grade, hazard
        else:
            # print(self.output_act(classifier))
            # return self.output_act(classifier)
            return classifier


class SelfAttentionBi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttentionBi, self).__init__()

        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.root = math.sqrt(dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mod1, mod2):
        x = torch.stack((mod1, mod2), dim=1)
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        QK = torch.bmm(Q, K.transpose(1, 2))
        attention_matrix = self.softmax(QK/self.root)
        out = torch.bmm(attention_matrix, V)
        return out


class FusionModelBi(nn.Module):
    def __init__(self, args, dim_in, dim_out):
        super(FusionModelBi, self).__init__()
        self.fusion_type = args.fusion_type
        act_layer = define_act_layer(args.act_type)

        if self.fusion_type == "attention":
            self.attention_module = SelfAttentionBi(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, dim_out*2, args.hidden_units, act_layer, args.dropout)
        elif self.fusion_type == "fused_attention":
            self.attention_module = SelfAttentionBi(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, (dim_out+1)**2, args.hidden_units, act_layer, args.dropout)
        elif self.fusion_type == "kronecker":
            self.taskmodel = MultiTaskModel(
                args.task, (dim_in+1)**2, args.hidden_units, act_layer, args.dropout)
        elif self.fusion_type == "concatenation":
            self.taskmodel = MultiTaskModel(
                args.task, dim_in*2, args.hidden_units, act_layer, args.dropout)
        else:
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')

    def forward(self, vec1, vec2):

        if self.fusion_type == "attention":
            x = self.attention_module(vec1, vec2)
            x = x.view(x.shape[0], x.shape[1]*x.shape[2])

        elif self.fusion_type == "kronecker":
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            x = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
                start_dim=1)

        elif self.fusion_type == "fused_attention":
            vec1, vec2 = self.attention_module(
                vec1, vec2)[:, 0, :], self.attention_module(vec1, vec2)[:, 1, :]
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            x = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
                start_dim=1)
            print(x.shape)

        elif self.fusion_type == "concatenation":
            x = torch.cat((vec1, vec2), dim=1)

        else:
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')
        return self.taskmodel(x)
    
class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()

        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.root = math.sqrt(dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mod1, mod2, mod3):
        x = torch.stack((mod1, mod2 ,mod3), dim=1)
        Q = self.WQ(x) 
        K = self.WK(x) 
        V = self.WV(x) 

        QK = torch.bmm(Q, K.transpose(1, 2)) 
        attention_matrix = self.softmax(QK/self.root)
        out = torch.bmm(attention_matrix, V)
        return out

    
class FusionModel(nn.Module):
    def __init__(self, args, dim_in, dim_out):
        super(FusionModel, self).__init__()
        self.fusion_type = args.fusion_type
        act_layer = define_act_layer(args.act_type)
    
        if self.fusion_type == "attention":
            self.attention_module = SelfAttention(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, dim_out*3, args.hidden_units, act_layer, args.dropout)
            
        elif self.fusion_type == "fused_attention":
            self.attention_module = SelfAttention(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, (dim_out+1)**3, args.hidden_units, act_layer, args.dropout)
            
        elif self.fusion_type == "kronecker":
            self.taskmodel = MultiTaskModel(
                args.task, (dim_in+1)**3, args.hidden_units, act_layer, args.dropout)
            
        elif self.fusion_type == "concatenation":
            self.taskmodel = MultiTaskModel(
                args.task, dim_in*3, args.hidden_units, act_layer, args.dropout)
            
        else:
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')
        
    def forward(self, vec1, vec2, vec3):
        
        if self.fusion_type == "attention":
            x = self.attention_module(vec1, vec2, vec3)
            x = x.view(x.shape[0], x.shape[1]*x.shape[2])
            
        elif self.fusion_type == "kronecker":
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            vec3 = torch.cat(
                (vec3, torch.ones((vec3.shape[0], 1)).to(vec3.device)), 1)
            x12 = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
            start_dim=1)
            x = torch.bmm(x12.unsqueeze(2), vec3.unsqueeze(1)).flatten(
                start_dim=1)
            
        elif self.fusion_type == "fused_attention":
            vec1, vec2, vec3 = self.attention_module(
                vec1, vec2, vec3)[:, 0, :], self.attention_module(vec1, vec2, vec3)[:, 1, :] , self.attention_module(vec1, vec2, vec3)[:, 2, :]
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            vec3 = torch.cat(
                (vec3, torch.ones((vec3.shape[0], 1)).to(vec3.device)), 1)
            x12 = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
                start_dim=1)
            x = torch.bmm(x12.unsqueeze(2), vec3.unsqueeze(1)).flatten(
                start_dim=1)
            
        elif self.fusion_type == "concatenation":
            x = torch.cat((vec1, vec2, vec3), dim=1)

        else: 
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')
        return self.taskmodel(x)

    

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.attention = nn.MultiheadAttention(emb_size, num_heads)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(tensor, "b n (h d) -> b n h d", h=self.num_heads) for tensor in qkv]
        q = rearrange(q, "b n h d -> (b h) n d")
        k = rearrange(k, "b n h d -> (b h) n d")
        v = rearrange(v, "b n h d -> (b h) n d")
        attn_output, _ = self.attention(q, k, v)
        attn_output = rearrange(attn_output, "(b h) n d -> b n (h d)", h=self.num_heads)
        return attn_output

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.projection = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)
        self.linear = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        x = self.projection(x)  # (B, emb_size, D/P, H/P, W/P)
        x = self.flatten(x)
        return self.linear(x.transpose(-1, -2))  # (B, num_patches, emb_size)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_ratio=4.0, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, int(emb_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(emb_size * mlp_ratio), emb_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x is expected to be of shape (num_patches, batch_size, emb_size)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, depth, num_heads, mlp_ratio=4.0, dropout_rate=0.1):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, mlp_ratio, dropout_rate) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pool(x.transpose(-1, -2)).squeeze(-1)
        return x
    
# class SwinTransformerRadiologyModel(nn.Module):

#     def __init__(
#         self,
#         patch_size: Union[Sequence[int], int],
#         window_size: Union[Sequence[int], int],
#         in_channels: int,
#         out_channels: int,
#         depths: Sequence[int] = (2, 2, 2, 2),
#         num_heads: Sequence[int] = (3, 6, 12, 24),
#         #try different feature_size!!
#         feature_size: int = 24,
#         norm_name: Union[Tuple, str] = "instance",
#         drop_rate: float = 0.7,
#         attn_drop_rate: float = 0.,
#         dropout_path_rate: float = 0.0,
#         normalize: bool = True,
#         use_checkpoint: bool = False,
#         spatial_dims: int = 3,
#         downsample="merging",
#     ) -> None:
#         """
#         Input requirement : [BxCxDxHxW]
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             feature_size: dimension of network feature size.
#             depths: number of layers in each stage.
#             num_heads: number of attention heads.
#             norm_name: feature normalization type and arguments.
#             drop_rate: dropout rate.
#             attn_drop_rate: attention dropout rate.
#             dropout_path_rate: drop path rate.
#             normalize: normalize output intermediate features in each stage.
#             use_checkpoint: use gradient checkpointing for reduced memory usage.
#             spatial_dims: number of spatial dims.
#             downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
#                 user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
#                 The default is currently `"merging"` (the original version defined in v0.9.0).


#         """

#         super().__init__()

#         if not (spatial_dims == 2 or spatial_dims == 3):
#             raise ValueError("spatial dimension should be 2 or 3.")

#         self.normalize = normalize

#         self.swinViT = SwinTransformer(
#             in_chans=in_channels,
#             embed_dim=feature_size,
#             window_size=window_size,
#             patch_size=patch_size,
#             depths=depths,
#             num_heads=num_heads,
#             mlp_ratio=4.0,
#             qkv_bias=True,
#             drop_rate=drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=dropout_path_rate,
#             norm_layer=nn.LayerNorm,
#             use_checkpoint=use_checkpoint,
#             spatial_dims=spatial_dims,
#             downsample=look_up_option(downsample) if isinstance(
#                 downsample, str) else downsample,
#         )
#         self.norm = nn.LayerNorm(feature_size*16)
#         self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
#         self.dim_reduction = nn.Conv3d(feature_size*16, out_channels, 1)
        
#     def forward(self, x_in):
#         hidden_states_out = self.swinViT(x_in, self.normalize)
#         hidden_output = rearrange(
#             hidden_states_out[4], "b c d h w -> b d h w c")
#         nomalized_hidden_states_out = self.norm(hidden_output)
#         nomalized_hidden_states_out = rearrange(
#             nomalized_hidden_states_out, "b d h w c -> b c d h w")
#         output = self.avgpool(nomalized_hidden_states_out)
#         output = torch.flatten(self.dim_reduction(output), 1)
#         # print(output.shape)

#         return output



# class CNNRadiologyModel(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         feature_size: int = 24,
#         spatial_dims: int = 3,
#         dropout_rate: float = 0.7,
#     ) -> None:
#         """
#         A simple 3D CNN-based feature extractor.
#         Input requirement : [BxCxDxHxW]
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             feature_size: dimension of network feature size.
#             spatial_dims: number of spatial dims (2 or 3).
#             dropout_rate: dropout rate.
#         """
#         super().__init__()

#         if spatial_dims != 3:
#             raise ValueError("This implementation is designed for 3D inputs.")

#         self.conv1 = nn.Conv3d(in_channels, feature_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(feature_size, feature_size * 2, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv3d(feature_size * 2, feature_size * 4, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv3d(feature_size * 4, feature_size * 8, kernel_size=3, padding=1)

#         self.bn1 = nn.BatchNorm3d(feature_size)
#         self.bn2 = nn.BatchNorm3d(feature_size * 2)
#         self.bn3 = nn.BatchNorm3d(feature_size * 4)
#         self.bn4 = nn.BatchNorm3d(feature_size * 8)

#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Conv3d(feature_size * 8, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.dropout(x)
#         x = self.pool(x)
#         x = torch.flatten(self.fc(x), 1)
#         #print(x.shape)
#         return x

class VisionTransformerRadiologyModel(nn.Module):
    def __init__(
        self,
        patch_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        emb_size: int,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.vit = VisionTransformer(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate
        )
        self.fc = nn.Linear(emb_size, out_channels)

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # self.extractor_ct_tumor = SwinTransformerRadiologyModel(
        #     patch_size=(1, 2, 2),
        #     window_size=[[4, 4, 4], [4, 4, 4], [8, 8, 8], [4, 4, 4]],
        #     in_channels=4,
        #     out_channels=args.feature_size,
        #     depths=(2, 2, 2, 2),
        #     num_heads=(3, 6, 12, 24),
        #     feature_size=int(args.feature_size/2),
        #     norm_name="instance",
        #     drop_rate=0.7,
        #     attn_drop_rate=0.,
        #     dropout_path_rate=0.2,
        #     normalize=True,
        #     use_checkpoint=False,
        #     spatial_dims=3
        # )
        # self.extractor_ct_lymph = SwinTransformerRadiologyModel(
        #     patch_size=(1, 2, 2),
        #     window_size=[[4, 4, 4], [4, 4, 4], [8, 8, 8], [4, 4, 4]],
        #     in_channels=4,
        #     out_channels=args.feature_size,
        #     depths=(2, 2, 2, 2),
        #     num_heads=(3, 6, 12, 24),
        #     feature_size=int(args.feature_size/2),
        #     norm_name="instance",
        #     drop_rate=0.7,
        #     attn_drop_rate=0.,
        #     dropout_path_rate=0.2,
        #     normalize=True,
        #     use_checkpoint=False,
        #     spatial_dims=3
        # )
        # self.extractor_ct_tumor = CNNRadiologyModel(
        #     in_channels=4, 
        #     out_channels=args.feature_size
        # )
        # self.extractor_ct_lymph = CNNRadiologyModel(
        #     in_channels=4, 
        #     out_channels=args.feature_size
        # )

        self.extractor_ct_tumor = VisionTransformerRadiologyModel(
            patch_size=(2, 2, 2),
            in_channels=4,
            out_channels=args.feature_size,
            emb_size=int(args.feature_size/2),
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout_rate=0.1
        )
        self.extractor_ct_lymph = VisionTransformerRadiologyModel(
            patch_size=(2, 2, 2),
            in_channels=4,
            out_channels=args.feature_size,
            emb_size=int(args.feature_size/2),
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout_rate=0.1
        )

        self.fusion = FusionModelBi(args, args.feature_size, args.dim_out)
        
    def forward(self, ct_tumor, ct_lymph):
        features_tumor = self.extractor_ct_tumor(ct_tumor)
        features_lymph = self.extractor_ct_lymph(ct_lymph)
        
        output = self.fusion(features_tumor, features_lymph)
        return output
        
