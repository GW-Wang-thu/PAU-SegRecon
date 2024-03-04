import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernelsize=(3, 3)):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernelsize, stride=(1, 1), padding=(kernelsize[0]//2, kernelsize[1]//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernelsize, stride=(1, 1), padding=(kernelsize[0]//2, kernelsize[1]//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.in_channels == self.out_channels:
            out = F.relu(self.bn2(x + self.conv2(out)))
        else:
            out = F.relu(self.bn2(self.conv2(out)))
        return out


class BasicBlock_light(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock_light, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            out = F.relu(self.bn1(self.conv1(x) + x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        return out


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size, mlp_inter_size, mhsa_n_heads, ):
        super(BasicTransformerBlock, self).__init__()

        # 前馈网络FeedForwardNetwork//(多层感知机MultiLayer Perceptron)
        self.FFN_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, mlp_inter_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_inter_size, hidden_size),
            nn.Dropout(0.1)
        )

        # 多头注意力Multi-Head Self Attention
        self.MHSA_all_head_size = mhsa_n_heads * (hidden_size // mhsa_n_heads)     # 头数 * 单头尺寸
        self.MHSA_n_heads = mhsa_n_heads
        self.MHSA_single_head_size = hidden_size // mhsa_n_heads
        self.MHSA_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.MHSA_query = nn.Linear(hidden_size, self.MHSA_all_head_size)
        self.MHSA_key = nn.Linear(hidden_size, self.MHSA_all_head_size)
        self.MHSA_value = nn.Linear(hidden_size, self.MHSA_all_head_size)
        self.MHSA_out_layer = nn.Linear(hidden_size, hidden_size)
        self.MHSA_attn_dropout = nn.Dropout(0.0)
        self.MHSA_proj_dropout = nn.Dropout(0.0)
        self.MHSA_softmax = nn.Softmax(dim=-1)

    def mhsa(self, x):
        normal_x = self.MHSA_norm(x)
        mixed_query = self.MHSA_query(normal_x)
        mixed_key = self.MHSA_key(normal_x)
        mixed_value = self.MHSA_value(normal_x)

        query = self.transpose_qkv(mixed_query)
        key = self.transpose_qkv(mixed_key)
        value = self.transpose_qkv(mixed_value)

        attention_score_book = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.MHSA_single_head_size)
        attention_score_book = self.MHSA_softmax(attention_score_book)
        attention_score_book= self.MHSA_attn_dropout(attention_score_book)

        context_value = torch.matmul(attention_score_book, value).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_value.size()[:-2] + (self.MHSA_all_head_size,)
        context_value = context_value.view(*new_context_layer_shape)
        attention_output = self.MHSA_out_layer(context_value)
        attention_output = self.MHSA_proj_dropout(attention_output)
        return attention_output

    def mlp(self, x):
        normal_x = self.FFN_norm(x)
        ffn_output = self.FFN(normal_x)
        return ffn_output

    def transpose_qkv(self, x):
        new_x_shape = x.size()[:-1] + (self.MHSA_n_heads, self.MHSA_single_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, latent):
        residue = self.mhsa(latent)
        latent = latent + residue
        residue = self.mlp(latent)
        latent = residue + latent
        return latent


class BasicTransformerBlock_NormalFree(nn.Module):
    def __init__(self, hidden_size, mlp_inter_size, mhsa_n_heads, ):
        super(BasicTransformerBlock_NormalFree, self).__init__()

        # 前馈网络FeedForwardNetwork//(多层感知机MultiLayer Perceptron)
        # self.FFN_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, mlp_inter_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_inter_size, hidden_size),
            nn.Dropout(0.1)
        )

        # 多头注意力Multi-Head Self Attention
        self.MHSA_all_head_size = mhsa_n_heads * (hidden_size // mhsa_n_heads)     # 头数 * 单头尺寸
        self.MHSA_n_heads = mhsa_n_heads
        self.MHSA_single_head_size = hidden_size // mhsa_n_heads
        # self.MHSA_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.MHSA_query = nn.Linear(hidden_size, self.MHSA_all_head_size)
        self.MHSA_key = nn.Linear(hidden_size, self.MHSA_all_head_size)
        self.MHSA_value = nn.Linear(hidden_size, self.MHSA_all_head_size)
        self.MHSA_out_layer = nn.Linear(hidden_size, hidden_size)
        self.MHSA_attn_dropout = nn.Dropout(0.0)
        self.MHSA_proj_dropout = nn.Dropout(0.0)
        self.MHSA_softmax = nn.Softmax(dim=-1)

    def mhsa(self, x):
        # normal_x = self.MHSA_norm(x)
        normal_x = x
        mixed_query = self.MHSA_query(normal_x)
        mixed_key = self.MHSA_key(normal_x)
        mixed_value = self.MHSA_value(normal_x)

        query = self.transpose_qkv(mixed_query)
        key = self.transpose_qkv(mixed_key)
        value = self.transpose_qkv(mixed_value)

        attention_score_book = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.MHSA_single_head_size)
        attention_score_book = self.MHSA_softmax(attention_score_book)
        attention_score_book= self.MHSA_attn_dropout(attention_score_book)

        context_value = torch.matmul(attention_score_book, value).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_value.size()[:-2] + (self.MHSA_all_head_size,)
        context_value = context_value.view(*new_context_layer_shape)
        attention_output = self.MHSA_out_layer(context_value)
        attention_output = self.MHSA_proj_dropout(attention_output)
        return attention_output

    def mlp(self, x):
        # normal_x = self.FFN_norm(x)
        normal_x = x
        ffn_output = self.FFN(normal_x)
        return ffn_output

    def transpose_qkv(self, x):
        new_x_shape = x.size()[:-1] + (self.MHSA_n_heads, self.MHSA_single_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, latent):
        residue = self.mhsa(latent)
        latent = latent + residue
        residue = self.mlp(latent)
        latent = residue + latent
        return latent


class Transformer(nn.Module):
    def __init__(self, latent_size, z_grid_size=8, y_grid_size=2, num_transformer_layers=16, hidden_length=256, LN=True):
        '''
        :param latent_size: 下采样latent的尺寸(通道数，高，宽)
        :param z_grid_size: 图像分块数量@深度维度
        :param y_grid_size: 图像分块数量@宽度维度
        '''
        super(Transformer, self).__init__()
        # Embeddings
        patch_size = (latent_size[1] // z_grid_size, latent_size[2] // y_grid_size)
        patch_size_conv = (patch_size[0], patch_size[1])
        n_patches = (latent_size[1] // patch_size[0]) * (latent_size[2] // patch_size[1])
        self.patch_embedding_layer = nn.Conv2d(in_channels=latent_size[0],
                                          out_channels=hidden_length,
                                          kernel_size=patch_size_conv,
                                          stride=patch_size_conv)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_length))
        self.embedding_dropout = nn.Dropout(0.1)

        # Encoders
        self.encoder = nn.Sequential()
        for _ in range(num_transformer_layers):
            if LN:
                self.encoder.append(BasicTransformerBlock(hidden_size=hidden_length, mlp_inter_size=512, mhsa_n_heads=16))
            else:
                self.encoder.append(BasicTransformerBlock_NormalFree(hidden_size=hidden_length, mlp_inter_size=512, mhsa_n_heads=16))
        self.encoder.append(nn.LayerNorm(hidden_length, eps=1e-6))

    # embedding
    def embedding(self, latent):
        emb_p = self.patch_embedding_layer(latent)  # B, hidden_length, grid_z, grid_y
        emb_p = emb_p.flatten(2)        # B, hidden_length, n_patches
        emb_p = emb_p.transpose(-1, -2)     # B, n_patches, hidden_length
        emb_pnp = emb_p + self.position_embeddings
        emb_pnp = self.embedding_dropout(emb_pnp)
        return emb_pnp

    def forward(self, latent):
        embeddings = self.embedding(latent)
        encoded = self.encoder(embeddings)
        return encoded


class UT_SegTransUNet_Normal(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[64, 128, 256, 512], num_class=5, grid_size=(32, 8), img_size=[512, 61], hidden_size=512):
        super(UT_SegTransUNet_Normal, self).__init__()
        # 下采样
        self.ResEncoderBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(medium_channels[0]),
            nn.ReLU(),
            BasicBlock(medium_channels[0], medium_channels[0]),
        )
        self.ResEncoderBlock2 = nn.Sequential(
            nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[1]),
            nn.ReLU(),
            BasicBlock(medium_channels[1], medium_channels[1]),
        )
        self.ResEncoderBlock3 = nn.Sequential(
            nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            BasicBlock(medium_channels[2], medium_channels[2]),
        )
        self.ResEncoderBlock4 = nn.Sequential(
            nn.Conv2d(medium_channels[2], medium_channels[2] * 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(medium_channels[2] * 4),
            nn.ReLU(),
        )
        self.vit = Transformer(latent_size=(256 * 4, 64, 16), hidden_length=hidden_size,  z_grid_size=grid_size[0], y_grid_size=grid_size[1])

        self.ResDecoderBlock0 = nn.Sequential(
            nn.Conv2d(hidden_size, medium_channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.ResDecoderBlock1 = nn.Sequential(
            BasicBlock(medium_channels[3], medium_channels[2]),
            nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock2 = nn.Sequential(
            BasicBlock(medium_channels[2], medium_channels[1]),
            nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock3 = nn.Sequential(
            BasicBlock(medium_channels[1], medium_channels[0]),
            nn.Conv2d(medium_channels[0], num_class, kernel_size=(3, 3), padding=1),
            nn.UpsamplingBilinear2d(size=img_size)
        )
        self.grid_size = grid_size

    def forward(self, x):
        skip_feature_1 = self.ResEncoderBlock1(x)
        skip_feature_2 = self.ResEncoderBlock2(skip_feature_1)
        skip_feature_3 = self.ResEncoderBlock3(skip_feature_2)
        encoded_feature = self.ResEncoderBlock4(skip_feature_3)
        trans_feature = self.vit(encoded_feature)
        B, n_patch, hidden = trans_feature.size()
        h, w = self.grid_size[0], self.grid_size[1]
        trans_feature_reshaped = trans_feature.permute(0, 2, 1)
        trans_feature_reshaped = trans_feature_reshaped.contiguous().view(B, hidden, h, w)
        trans_feature_us = self.ResDecoderBlock0(trans_feature_reshaped)
        up_feature_1 = self.ResDecoderBlock1(torch.concat([skip_feature_3, trans_feature_us], dim=1))
        up_feature_2 = self.ResDecoderBlock2(torch.concat([skip_feature_2, up_feature_1], dim=1))
        output = self.ResDecoderBlock3(torch.concat([skip_feature_1, up_feature_2], dim=1))
        return output


class UT_SegTransUNet_tiny(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[16, 32, 64, 128], num_class=5, grid_size=(32, 8), img_size=[512, 61], hidden_size=128):
        super(UT_SegTransUNet_tiny, self).__init__()
        # 下采样
        self.ResEncoderBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(medium_channels[0]),
            nn.ReLU(),
            BasicBlock(medium_channels[0], medium_channels[0]),
        )
        self.ResEncoderBlock2 = nn.Sequential(
            nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[1]),
            nn.ReLU(),
            BasicBlock(medium_channels[1], medium_channels[1]),
        )
        self.ResEncoderBlock3 = nn.Sequential(
            nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            BasicBlock(medium_channels[2], medium_channels[2]),
        )
        self.ResEncoderBlock4 = nn.Sequential(
            nn.Conv2d(medium_channels[2], medium_channels[2] * 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(medium_channels[2] * 4),
            nn.ReLU(),
        )
        self.vit = Transformer(latent_size=(medium_channels[2] * 4, 64, 16), hidden_length=hidden_size, z_grid_size=grid_size[0],
                               y_grid_size=grid_size[1])

        self.ResDecoderBlock0 = nn.Sequential(
            nn.Conv2d(hidden_size, medium_channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.ResDecoderBlock1 = nn.Sequential(
            BasicBlock(medium_channels[3], medium_channels[2]),
            nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock2 = nn.Sequential(
            BasicBlock(medium_channels[2], medium_channels[1]),
            nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock3 = nn.Sequential(
            BasicBlock(medium_channels[1], medium_channels[0]),
            nn.Conv2d(medium_channels[0], num_class, kernel_size=(3, 3), padding=1),
            nn.UpsamplingBilinear2d(size=img_size)
        )
        self.grid_size = grid_size

    def forward(self, x):
        skip_feature_1 = self.ResEncoderBlock1(x)
        skip_feature_2 = self.ResEncoderBlock2(skip_feature_1)
        skip_feature_3 = self.ResEncoderBlock3(skip_feature_2)
        encoded_feature = self.ResEncoderBlock4(skip_feature_3)
        trans_feature = self.vit(encoded_feature)
        B, n_patch, hidden = trans_feature.size()
        h, w = self.grid_size[0], self.grid_size[1]
        trans_feature_reshaped = trans_feature.permute(0, 2, 1)
        trans_feature_reshaped = trans_feature_reshaped.contiguous().view(B, hidden, h, w)
        trans_feature_us = self.ResDecoderBlock0(trans_feature_reshaped)
        up_feature_1 = self.ResDecoderBlock1(torch.concat([skip_feature_3, trans_feature_us], dim=1))
        up_feature_2 = self.ResDecoderBlock2(torch.concat([skip_feature_2, up_feature_1], dim=1))
        output = self.ResDecoderBlock3(torch.concat([skip_feature_1, up_feature_2], dim=1))
        return output


class UT_SegTransUNet_utiny(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[12, 24, 48, 96], num_class=5, grid_size=(32, 8), img_size=[512, 61], hidden_size=128):
        super(UT_SegTransUNet_utiny, self).__init__()
        # 下采样
        self.ResEncoderBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(medium_channels[0]),
            nn.ReLU(),
            BasicBlock(medium_channels[0], medium_channels[0]),
        )
        self.ResEncoderBlock2 = nn.Sequential(
            nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[1]),
            nn.ReLU(),
            BasicBlock(medium_channels[1], medium_channels[1]),
        )
        self.ResEncoderBlock3 = nn.Sequential(
            nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            BasicBlock(medium_channels[2], medium_channels[2]),
        )
        self.ResEncoderBlock4 = nn.Sequential(
            nn.Conv2d(medium_channels[2], medium_channels[2] * 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(medium_channels[2] * 4),
            nn.ReLU(),
        )
        self.vit = Transformer(latent_size=(medium_channels[2] * 4, 64, 16), hidden_length=hidden_size, z_grid_size=grid_size[0],
                               y_grid_size=grid_size[1])

        self.ResDecoderBlock0 = nn.Sequential(
            nn.Conv2d(hidden_size, medium_channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=(64, 16))
        )
        self.ResDecoderBlock1 = nn.Sequential(
            BasicBlock(medium_channels[3], medium_channels[2]),
            nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock2 = nn.Sequential(
            BasicBlock(medium_channels[2], medium_channels[1]),
            nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock3 = nn.Sequential(
            BasicBlock(medium_channels[1], medium_channels[0]),
            nn.Conv2d(medium_channels[0], num_class, kernel_size=(3, 3), padding=1),
            nn.UpsamplingBilinear2d(size=img_size)
        )
        self.grid_size = grid_size

    def forward(self, x):
        skip_feature_1 = self.ResEncoderBlock1(x)
        skip_feature_2 = self.ResEncoderBlock2(skip_feature_1)
        skip_feature_3 = self.ResEncoderBlock3(skip_feature_2)
        encoded_feature = self.ResEncoderBlock4(skip_feature_3)
        trans_feature = self.vit(encoded_feature)
        B, n_patch, hidden = trans_feature.size()
        h, w = self.grid_size[0], self.grid_size[1]
        trans_feature_reshaped = trans_feature.permute(0, 2, 1)
        trans_feature_reshaped = trans_feature_reshaped.contiguous().view(B, hidden, h, w)
        trans_feature_us = self.ResDecoderBlock0(trans_feature_reshaped)
        up_feature_1 = self.ResDecoderBlock1(torch.concat([skip_feature_3, trans_feature_us], dim=1))
        up_feature_2 = self.ResDecoderBlock2(torch.concat([skip_feature_2, up_feature_1], dim=1))
        output = self.ResDecoderBlock3(torch.concat([skip_feature_1, up_feature_2], dim=1))
        return output


class UT_SegTransUNet_light(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[32, 64, 128, 256], num_class=5, grid_size=(32, 8), img_size=[512, 61], hidden_size=256, LN=True):
        super(UT_SegTransUNet_light, self).__init__()
        # 下采样
        self.ResEncoderBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(medium_channels[0]),
            nn.ReLU(),
            BasicBlock(medium_channels[0], medium_channels[0]),
        )
        self.ResEncoderBlock2 = nn.Sequential(
            nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[1]),
            nn.ReLU(),
            BasicBlock(medium_channels[1], medium_channels[1]),
        )
        self.ResEncoderBlock3 = nn.Sequential(
            nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            BasicBlock(medium_channels[2], medium_channels[2]),
        )
        self.ResEncoderBlock4 = nn.Sequential(
            nn.Conv2d(medium_channels[2], medium_channels[2] * 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(medium_channels[2] * 4),
            nn.ReLU(),
        )
        self.vit = Transformer(latent_size=(medium_channels[2] * 4, 64, 16), hidden_length=hidden_size,
                               z_grid_size=grid_size[0],
                               y_grid_size=grid_size[1], LN=LN)

        self.ResDecoderBlock0 = nn.Sequential(
            nn.Conv2d(hidden_size, medium_channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=(64, 16))
        )
        self.ResDecoderBlock1 = nn.Sequential(
            BasicBlock(medium_channels[3], medium_channels[2]),
            nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock2 = nn.Sequential(
            BasicBlock(medium_channels[2], medium_channels[1]),
            nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock3 = nn.Sequential(
            BasicBlock(medium_channels[1], medium_channels[0]),
            nn.Conv2d(medium_channels[0], num_class, kernel_size=(3, 3), padding=1),
            nn.UpsamplingBilinear2d(size=img_size)
        )
        self.grid_size = grid_size

    def forward(self, x):
        skip_feature_1 = self.ResEncoderBlock1(x)
        skip_feature_2 = self.ResEncoderBlock2(skip_feature_1)
        skip_feature_3 = self.ResEncoderBlock3(skip_feature_2)
        encoded_feature = self.ResEncoderBlock4(skip_feature_3)
        trans_feature = self.vit(encoded_feature)
        B, n_patch, hidden = trans_feature.size()
        h, w = self.grid_size[0], self.grid_size[1]
        trans_feature_reshaped = trans_feature.permute(0, 2, 1)
        trans_feature_reshaped = trans_feature_reshaped.contiguous().view(B, hidden, h, w)
        trans_feature_us = self.ResDecoderBlock0(trans_feature_reshaped)
        up_feature_1 = self.ResDecoderBlock1(torch.concat([skip_feature_3, trans_feature_us], dim=1))
        up_feature_2 = self.ResDecoderBlock2(torch.concat([skip_feature_2, up_feature_1], dim=1))
        output = self.ResDecoderBlock3(torch.concat([skip_feature_1, up_feature_2], dim=1))
        return output


class UT_SegTransUNet_light_dense(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[32, 64, 128, 256], num_class=5, grid_size=(32, 8), img_size=[512, 61], hidden_size=256):
        super(UT_SegTransUNet_light_dense, self).__init__()
        # 下采样
        self.ResEncoderBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(medium_channels[0]),
            nn.ReLU(),
            BasicBlock(medium_channels[0], medium_channels[0]),
        )
        self.ResEncoderBlock2 = nn.Sequential(
            nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[1]),
            nn.ReLU(),
            BasicBlock(medium_channels[1], medium_channels[1]),
        )
        self.ResEncoderBlock3 = nn.Sequential(
            nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            BasicBlock(medium_channels[2], medium_channels[2]),
        )
        self.ResEncoderBlock4 = nn.Sequential(
            nn.Conv2d(medium_channels[2], medium_channels[2] * 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(medium_channels[2] * 4),
            nn.ReLU(),
        )
        self.vit = Transformer(latent_size=(medium_channels[2] * 4, 64, 16), hidden_length=hidden_size,
                               z_grid_size=grid_size[0],
                               y_grid_size=grid_size[1])
        # self.ResUpsample = nn.UpsamplingBilinear2d(size=(64, 16))
        self.ResDecoderBlock0 = nn.Sequential(
            nn.Conv2d(hidden_size, medium_channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(medium_channels[2]),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=(64, 16))
        )
        self.ResDecoderBlock1 = nn.Sequential(
            BasicBlock(medium_channels[3], medium_channels[2]),
            nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock2 = nn.Sequential(
            BasicBlock(medium_channels[2], medium_channels[1]),
            nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1,
                               output_padding=(1, 0)),
            nn.ReLU(),
        )
        self.ResDecoderBlock3 = nn.Sequential(
            BasicBlock(medium_channels[1], medium_channels[0]),
            nn.Conv2d(medium_channels[0], num_class, kernel_size=(3, 3), padding=1),
            nn.UpsamplingBilinear2d(size=img_size)
        )
        self.grid_size = grid_size

    def forward(self, x):
        skip_feature_1 = self.ResEncoderBlock1(x)
        skip_feature_2 = self.ResEncoderBlock2(skip_feature_1)
        skip_feature_3 = self.ResEncoderBlock3(skip_feature_2)
        encoded_feature = self.ResEncoderBlock4(skip_feature_3)
        trans_feature = self.vit(encoded_feature)
        B, n_patch, hidden = trans_feature.size()
        h, w = self.grid_size[0], self.grid_size[1]
        trans_feature_reshaped = trans_feature.permute(0, 2, 1)
        trans_feature_reshaped = trans_feature_reshaped.contiguous().view(B, hidden, h, w)
        trans_feature_us = self.ResDecoderBlock0(trans_feature_reshaped)
        up_feature_1 = self.ResDecoderBlock1(torch.concat([skip_feature_3, trans_feature_us], dim=1))
        up_feature_2 = self.ResDecoderBlock2(torch.concat([skip_feature_2, up_feature_1], dim=1))
        output = self.ResDecoderBlock3(torch.concat([skip_feature_1, up_feature_2], dim=1))
        return output

