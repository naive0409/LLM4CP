import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPVisionConfig, CLIPTextConfig
from einops import rearrange
from Embed import DataEmbedding, VisionEmbedding

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class Model(nn.Module):
    model_list = ["gpt2", "clip"]

    def __init__(self, gpt_type=model_list[1], d_ff=512, d_model=768, gpt_layers=6,  # clip
    # def __init__(self, gpt_type=model_list[0], d_ff=768, d_model=768, gpt_layers=6,  # gpt2
                 pred_len=4, prev_len=16, mlp=0, res_layers=4,
                 K=48, UQh=4, UQv=1, BQh=2, BQv=1,
                 patch_size=8, stride=1, res_dim=64,
                 embed='timeF', freq='h', dropout=0.1):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = d_ff
        self.d_model = d_model

        self.K = K
        self.UQh = UQh
        self.UQv = UQv
        self.BQh = BQh
        self.BQv = BQv
        self.Nt = UQh * UQv
        self.Nr = BQh * BQv
        self.mul = prev_len * K * UQh * UQv * BQh * BQv
        self.enc_in = K * UQh * UQv * BQh * BQv
        self.c_out = K * UQh * UQv * BQh * BQv

        self.enc_embedding1 = DataEmbedding(2 * self.enc_in, 512, embed, freq, dropout)
        self.enc_embedding2 = VisionEmbedding(image_size=[self.prev_len, 2 * self.enc_in], patch_size=self.patch_size)

        if gpt_type == 'gpt2-medium':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1024
        elif gpt_type == 'gpt2-large':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1280
        elif gpt_type == 'gpt2-xl':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1600
        elif gpt_type == 'clip':
            # done clip替换gpt2
            # done noTokenizer : hidden_states不用clip embeddings生成
            # vision_config = CLIPVisionConfig(num_channels=1)
            # text_config = CLIPTextConfig()
            # clip_config = CLIPConfig().from_text_vision_configs(text_config, vision_config)
            self.gpt2 = CLIPModel.from_pretrained("./models/openai-clip-vit-base-patch32")
                                                  # config=clip_config,
                                                  # ignore_mismatched_sizes=True
                                                  # )
            # self.gpt_dim = 512
            # self.clipVisionModel = CLIPVisionModel.from_pretrained("./models/openai-clip-vit-base-patch32")
            # self.clip_processor = CLIPProcessor.from_pretrained("./models/openai-clip-vit-base-patch32")
        else:
            self.gpt2 = GPT2Model.from_pretrained('./models/gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 768

        '''param.requires_grad: true-no frozen, false-frozen
        gpt2:
            wte:False,word token embedding词向量编码
            wpe:true,position embedding, token的位置编码
            ln:true,layer norm
            mlp:false
            att:false
            
        clip:
            logit_scale:true
            
            text_model.embeddings.token_embedding:false
            text_model.embeddings.position_embedding:true
            
            text_model.final_layer_norm.weight:true
            
            vision_model.embeddings.patch_embedding:false
            vision_model.embeddings.position_embedding:true
            vision_model.embeddings.class_embedding(198):true
            
            vision_model.pre_layrnorm:true
            vision_model.post_layernorm(395):true
            
            visual_projection(-2):true
            text_projection:true
            
            layer-norm:true
            mlp:false
            self_attn:false
        '''

        if gpt_type == 'gpt2':
            print('Model:gpt2')
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                    param.requires_grad = True
                elif 'mlp' in name and mlp == 1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif gpt_type == 'clip':
            print('Model:clip')
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'layer-norm' in name or 'layernorm' in name or 'layer_norm' in name:
                    param.requires_grad = True
                elif 'mlp' in name and mlp == 1:
                    param.requires_grad = True
                elif 'position_embedding' in name:
                    param.requires_grad = True
                elif 'class_embedding' in name:
                    param.requires_grad = True
                elif 'visual_projection' in name or 'text_projection' in name:
                    param.requires_grad = True
                elif 'logit_scale' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise ValueError(f'gpt_type {gpt_type} not supported')

        with open(f'./code_testing/structure_{gpt_type}.csv', 'w') as file:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                file.write(';'.join(str(x) for x in [i, param.requires_grad, list(param.data.shape), name]) + '\n')

        # if use_gpu:
        #     device = torch.device('cuda:{}'.format(gpu_id))
        #     self.gpt2.to(device=device)

        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        self.predict_linear_vision_pre = nn.Linear(1 + int(2 * self.enc_in * self.prev_len // (self.patch_size ** 2)),
                                                   1 + int(2 * self.enc_in * self.prev_len // (self.patch_size ** 2)))
        self.out_layer_dim = nn.Linear(d_ff, self.c_out * 2)
        self.output_layer_time = nn.Sequential(
            nn.Linear(self.prev_len, self.pred_len)
        )

        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(res_dim))
            self.RB_f.append(Res_block(res_dim))
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        B, L, enc_in = x_enc.shape  # [B, L, D]  x_enc:torch.Size([1024, 16, 96])
        # process in delay domain
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)  # torch.Size([1024, 16, 48, 2])
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])  # torch.Size([1024, 16, 48])
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)  # torch.Size([1024, 16, 48])
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)  # torch.Size([1024, 16, 96])
        x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in)  # torch.Size([1024, 4, 4, 96])
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # torch.Size([1024, 4, 4, 96])
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in)  # torch.Size([1024, 16, 96])
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)  # torch.Size([1024, 2, 16, 48])
        x_enc_delay = self.RB_f(x_enc_delay)  # torch.Size([1024, 2, 16, 48])
        # process in frequency domain
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)  # torch.Size([1024, 4, 4, 96])
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # torch.Size([1024, 4, 4, 96])
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)  # torch.Size([1024, 16, 96])
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)  # torch.Size([1024, 2, 16, 48])
        x_enc_fre = self.RB_e(x_enc_fre)  # torch.Size([1024, 2, 16, 48])

        # x_enc = x_enc_fre + x_enc_delay  # torch.Size([1024, 2, 16, 48])
        # x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)  # [B, L, D] torch.Size([1024, 16, 96])
        #
        # enc_out = self.enc_embedding1(x_enc, x_mark_enc)  # [B, L, 768] torch.Size([1024, 16, 768])

        # vision emb
        x_enc_delay = rearrange(x_enc_delay, 'b o l k -> b 1 l (k o)', o=2)  # torch.Size([1024, 16, 96])
        x_enc_delay = self.enc_embedding2(x_enc_delay)
        x_enc_delay = self.predict_linear_vision_pre(x_enc_delay.permute(0, 2, 1)).permute(0, 2, 1)
        # text emb
        x_enc_fre = rearrange(x_enc_fre, 'b o l k -> b l (k o)', o=2)  # torch.Size([1024, 16, 96])
        x_enc_fre = self.enc_embedding1(x_enc_fre, x_mark_enc)  # torch.Size([1024, 16, 512])
        x_enc_fre = self.predict_linear_pre(x_enc_fre.permute(0, 2, 1)).permute(0, 2, 1)

        dec_out = self.gpt2(input_ids=x_enc_fre, pixel_values=x_enc_delay)#.last_hidden_state  # [B, L, 768]
        # todo clip输出处理
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.out_layer_dim(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        dec_out = dec_out * std + mean

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

if __name__ == '__main__':
    import torch

    # device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(UQh=1, UQv=1, BQh=1, BQv=1).to(device)
    inputs = torch.rand(3, 16, 96).to(device)
    out = model(inputs, None, None, None)
    print(out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
