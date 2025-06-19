import os
import numpy as np
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import CLIPTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import CLIPModel, CLIPVisionModel, CLIPTextModel
from einops import rearrange
from Embed import DataEmbedding, VisionEmbedding

from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize

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


class MmFF_block(nn.Module):
    """
    多模态特征融合块(Multi-modal Feature Fusion Block)类定义。

    该块旨在处理来自不同模态的数据，并可选择性地将它们融合在一起。
    主要通过并行的Res_block对每个模态的数据进行处理，然后根据融合标志(fusion_flag)
    决定是否将辅助模态的特征加到主模态(下标为0)上。

    参数:
    - res_dim (int): 输入到每个Res_block的维度。
    - modality_num (int, optional): 模态的数量，默认为2。
    - fusion_flag (bool, optional): 是否进行特征融合，默认为True。
    """
    def __init__(self, res_dim: int, modality_num: int = 2, fusion_flag: bool = True):
        super(MmFF_block, self).__init__()

        self.modality_num = modality_num
        self.res_dim = res_dim
        self.fusion_flag = fusion_flag
        # 有多少模态就有多少并行的Res_block
        # Res_block_list[0]是主模态
        self.Res_block_list = nn.ModuleList([Res_block(in_planes=self.res_dim) for _ in range(self.modality_num)])

    def forward(self, x: tuple) -> list:
        x = list(x)
        assert len(x) == self.modality_num, "模态数量错误"

        for modality_i in range(self.modality_num):
            x[modality_i] = self.Res_block_list[modality_i](x[modality_i])

        if self.fusion_flag:
            for modality_j in range(self.modality_num):
                if modality_j != 0:
                    x[0] = torch.add(x[0], x[modality_j])
        return x


class MmHFF(nn.Module):
    """
    多模态多层次融合(Multi-modal Hierarchical Feature Fusion, MmHFF)类

    参数:
    - res_dim (int): 特征维度的大小，用于配置输入卷积层inConv2d_list和输出卷积层outConv2d_list
    - modality_num (int, 可选): 模态的数量，默认为2
    - fusion_flag_list (list, 可选): 融合标志列表，默认为[False, True, False, True]

    此构造函数初始化了MmHFF类，设置了模态数量、特征维度和融合标志列表，
    并创建了输入卷积层、输出卷积层和多模态融合块的序列。
    """
    def __init__(self, res_dim: int, modality_num: int = 2, fusion_flag_list: list = None):
        super(MmHFF, self).__init__()

        if fusion_flag_list is None:
            fusion_flag_list = [False, True, False, True]
        self.modality_num = modality_num
        self.res_dim = res_dim
        self.fusion_flag_list = fusion_flag_list
        self.MmFF_block_layers = len(self.fusion_flag_list)

        self.inConv2d_list = nn.ModuleList([nn.Conv2d(in_channels=2,
                                                      out_channels=res_dim,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
                                            for _ in range(self.modality_num)])
        self.outConv2d_list = nn.ModuleList([nn.Conv2d(in_channels=res_dim,
                                                       out_channels=2,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1)
                                             for _ in range(self.modality_num)])
        self.MmFF_block_list = nn.Sequential()
        for i in range(self.MmFF_block_layers):
            self.MmFF_block_list.add_module(f"MmFF_block_{i}",
                                            MmFF_block(res_dim=self.res_dim,
                                                       modality_num=self.modality_num,
                                                       fusion_flag=self.fusion_flag_list[i]))

    def forward(self, x: tuple) -> torch.Tensor:
        x = list(x)
        assert len(x) == self.modality_num, "模态数量错误"
        for modality_i in range(self.modality_num):
            x[modality_i] = self.inConv2d_list[modality_i](x[modality_i])

        x = self.MmFF_block_list(x)

        for modality_i in range(self.modality_num):
            x[modality_i] = self.outConv2d_list[modality_i](x[modality_i])

        for modality_i in range(self.modality_num):
            if modality_i != 0:
                x[0] = torch.add(x[0], x[modality_i])
        return x[0]


class Model(nn.Module):
    model_list = ["gpt2", "clip", "clip_vision", "clip_text"]

    # def __init__(self, gpt_type=model_list[3], d_ff=512, d_model=512, gpt_layers=6,  # done clip text
    # def __init__(self, gpt_type=model_list[2], d_ff=768, d_model=768, gpt_layers=6,  # done clip vision
    def __init__(self, gpt_type=model_list[0], d_ff=768, d_model=768, gpt_layers=6,  # done gpt2
                 pred_len=4, prev_len=16, mlp=0, res_layers=4,
                 K=48, UQh=4, UQv=1, BQh=2, BQv=1,
                 patch_size=4, stride=2, res_dim=64,
                 embed='timeF', freq='h', dropout=0.1):
        super(Model, self).__init__()
        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = d_ff
        self.d_model = d_model
        self.n_heads = 8
        self.enc_in_ = 7
        self.seq_len = 96
        self.num_tokens = 64

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

        self.enc_embedding1 = DataEmbedding(2 * self.enc_in, self.d_model, embed, freq, dropout)
        # self.enc_embedding2 = VisionEmbedding(image_size=[self.prev_len, 2 * self.enc_in], patch_size=self.patch_size)


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
            self.gpt2 = CLIPModel.from_pretrained("./models/openai-clip-vit-base-patch32")
        elif gpt_type == 'clip_vision':
            self.gpt2 = CLIPVisionModel.from_pretrained("./models/openai-clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("./models/openai-clip-vit-base-patch32")
        elif gpt_type == 'clip_text':
            self.gpt2 = CLIPTextModel.from_pretrained("./models/openai-clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("./models/openai-clip-vit-base-patch32")

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
        elif gpt_type == 'clip' or gpt_type == 'clip_vision' or gpt_type == 'clip_text':
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

        self.description = 'Channel state information (CSI) plays a fundamental rolein facilitating m-MIMO related design.'

        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_size, self.stride, dropout)

        '''
        self.word_embeddings = self.gpt2.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_ff, 512)#self.d_llm)
        '''
        self.patch_nums = int((self.seq_len - self.patch_size) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        # self.output_projection = FlattenHead(self.enc_in_, self.head_nf, self.pred_len, head_dropout=dropout)
        # self.output_projection = FlattenHead(self.enc_in_, self.head_nf, 96, head_dropout=dropout)
        # self.normalize_layers = Normalize(self.enc_in_, affine=False)

        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        # self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.final_dim = int((self.prev_len - self.patch_size) / self.stride + 1) * self.patch_size
        # self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        self.predict_linear_pre = nn.Linear(self.final_dim, self.final_dim)
        # self.vision_features = 1 + int(2 * self.enc_in * self.prev_len // (self.patch_size ** 2))
        # self.output_layer_time = nn.Linear(self.prev_len, self.pred_len)
        # self.output_layer_time = nn.Linear(self.final_dim, self.pred_len)

        '''
        self.vision_features是用kernel_size=patch_size,stride=patch_size的conv2d层
        处理尺寸为(2 * self.enc_in) * self.prev_len的x_enc_delay后
        结果的维数.
        例如：enc_in = 48, prev_len = 16, patch_size=4,
        则self.vision_features = 2*48/4 * 16/4 + 1= 97
        self.predict_linear_vision_pre = nn.Linear(self.vision_features, self.vision_features)

        self.down_layer_vision_dim = nn.Linear(768, 512)
        self.down_layer_vision_time = nn.Linear(self.vision_features, self.prev_len)
        '''
        # '''
        # clip的vision输出为[8, 97, 768]，需要变换到[8, 16, 512]，和text输出相同
        # 先下降dim再下降time
        self.out_layer_dim = nn.Linear(self.d_ff, self.c_out * 2)
        # self.output_layer_time = nn.Sequential(
        #     nn.Linear(self.prev_len, self.pred_len)
        # )
        self.output_layer_time = nn.Linear(self.final_dim, self.pred_len)

        # '''
        # '''
        # self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        # self.RB_f = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        # for i in range(self.res_layers):
        #     self.RB_e.append(Res_block(res_dim))
        #     self.RB_f.append(Res_block(res_dim))
        # self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        # self.RB_f.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        # '''

        self.MmHFF = MmHFF(res_dim=res_dim, modality_num=2, fusion_flag_list=[False, True, False, True])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        # x_enc = x_enc.permute(0, 2, 1)
        # timellm里的顺序是batch_size\Time Steps\number of features，这里先调换过来

        # x_enc = self.normalize_layers(x_enc, 'norm')
        # '''
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        # '''


        B, L, enc_in = x_enc.shape  # [B, L, D]  x_enc:torch.Size([1024, 16, 96])
        # B T N -> B * N , T , 1
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * enc_in, L, 1)

        # min_values = torch.min(x_enc, dim=1)[0]
        # max_values = torch.max(x_enc, dim=1)[0]
        # medians = torch.median(x_enc, dim=1).values
        # lags = self.calcute_lags(x_enc)
        # trends = x_enc.diff(dim=1).sum(dim=1)

        # prompt = []
        # for b in range(x_enc.shape[0]):
        #     min_values_str = str(min_values[b].tolist()[0])
        #     max_values_str = str(max_values[b].tolist()[0])
        #     median_values_str = str(medians[b].tolist()[0])
        #     # lags_values_str = str(lags[b].tolist())
        #     prompt_ = (
        #         f"<|start_prompt|>Dataset description: {self.description}"
        #         f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.prev_len)} steps information; "
        #         # "Input statistics: "
        #         # f"min value {min_values_str}, "
        #         # f"max value {max_values_str}, "
        #         # f"median value {median_values_str}, "
        #         f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}. "
        #         # f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
        #     )
        #
        #     prompt.append(prompt_)
        #
        # x_enc = x_enc.reshape(B, enc_in, L).permute(0, 2, 1).contiguous()
        #
        # prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # x_enc = x_enc.permute(0, 2, 1).contiguous()
        # x_enc = x_enc.to(torch.bfloat16)  # 临时加的
        # enc_out, n_vars = self.patch_embedding(x_enc)  # todo bfloat16?
        # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state

        # clip_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # enc_out = self.RB_e(enc_out)  # torch.Size([1024, 2, 16, 48])
        # clip_enc_out = enc_out

        # '''
        # process in delay domain
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)  # torch.Size([1024, 16, 48, 2])
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])  # torch.Size([1024, 16, 48])
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)  # torch.Size([1024, 16, 48])
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)  # torch.Size([1024, 16, 96])
        # x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in)  # torch.Size([1024, 4, 4, 96])
        x_enc_delay = x_enc_delay.unfold(dimension=-2, size=self.patch_size, step=self.stride)
        x_enc_delay = x_enc_delay.permute(0, 1, 3, 2)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # torch.Size([1024, 4, 4, 96])
        x_enc_delay = x_enc_delay.reshape(B, -1, enc_in)  # torch.Size([1024, 16, 96])
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)  # torch.Size([1024, 2, 16, 48])
        # x_enc_delay = self.RB_f(x_enc_delay)  # torch.Size([1024, 2, 16, 48])
        # process in frequency domain
        # x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)  # torch.Size([1024, 4, 4, 96])
        x_enc_fre = x_enc.unfold(dimension=-2, size=self.patch_size, step=self.stride)
        x_enc_fre = x_enc_fre.permute(0, 1, 3, 2)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # torch.Size([1024, 4, 4, 96])
        x_enc_fre = x_enc_fre.reshape(B, -1, enc_in)  # torch.Size([1024, 16, 96])
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)  # torch.Size([1024, 2, 16, 48])
        # x_enc_fre = self.RB_e(x_enc_fre)  # torch.Size([1024, 2, 16, 48])

        # x_enc = x_enc_fre + x_enc_delay  # torch.Size([1024, 2, 16, 48])
        x_enc = (x_enc_fre, x_enc_delay)

        x_enc = self.MmHFF(x_enc)

        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)  # [B, L, D] torch.Size([1024, 16, 96])

        enc_out = self.enc_embedding1(x_enc, x_mark_enc)  # [B, L, 768] torch.Size([1024, 16, 768])

        # # vision emb
        # x_enc_delay = rearrange(x_enc_delay, 'b o l k -> b 1 l (k o)', o=2)  # torch.Size([1024, 16, 96])
        # x_enc_delay = self.enc_embedding2(x_enc_delay)
        # x_enc_delay = self.predict_linear_vision_pre(x_enc_delay.permute(0, 2, 1)).permute(0, 2, 1)
        # # text emb
        # x_enc_fre = rearrange(x_enc_fre, 'b o l k -> b l (k o)', o=2)  # torch.Size([1024, 16, 96])
        # x_enc_fre = self.enc_embedding1(x_enc_fre, x_mark_enc)  # torch.Size([1024, 16, 512])
        # x_enc_fre = self.predict_linear_pre(x_enc_fre.permute(0, 2, 1)).permute(0, 2, 1)

        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # enc_out = torch.nn.functional.pad(enc_out, (0, self.gpt_dim - enc_out.shape[-1]))
        # '''

        # dec_out = self.gpt2(input_ids=x_enc_fre, pixel_values=x_enc_delay, return_loss=True)
        # dec_out = self.gpt2(pixel_values=enc_out)  # done clip
        # dec_out = self.gpt2(input_ids=clip_enc_out)  # done clip text
        # dec_out = self.gpt2(input_ids=enc_out)  # done clip text
        # dec_out = self.gpt2(pixel_values=enc_out)  # done clip vision
        dec_out = self.gpt2(inputs_embeds=enc_out)#.last_hidden_state  # done gpt2 [B , L, 768]
        # clip_loss = dec_out.loss

        # todo clip输出处理
        # dec_out_text = dec_out.text_model_output.last_hidden_state  # [B, L, 512]
        # dec_out_vision = dec_out.vision_model_output.last_hidden_state  # [B, 1 + 16/patch_size * 96/patch_size, 768]
        dec_out = dec_out.last_hidden_state  # done clip & clipvision & clip text [B, L, 512]
        dec_out = dec_out[:, :, :self.d_ff]  # 128 可变 512

        # dec_out_vision = self.down_layer_vision_dim(dec_out_vision)
        # dec_out_vision = self.down_layer_vision_time(dec_out_vision.permute(0, 2, 1)).permute(0, 2, 1)

        # dec_out = dec_out_vision + dec_out_text
        # '''
        dec_out = self.out_layer_dim(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        dec_out = dec_out * std + mean
        # '''
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        # dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = torch.reshape(dec_out, (-1, 16, dec_out.shape[-2], dec_out.shape[-1]))
        # dec_out = dec_out.permute(0, 2, 1).contiguous()  # 8,16，512 可变

        # dec_out = dec_out[:, -self.pred_len:, :, -self.patch_nums:]
        # dec_out = dec_out[:, :, :, -self.patch_nums:]

        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  # 8,16，512 可变
        dec_out = self.output_projection(dec_out)

        '''
        dec_out = self.out_layer_dim(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = torch.reshape(dec_out, (dec_out.shape[0], -1))

        dec_out = self.out_layer_dim_2(dec_out)#.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = torch.reshape(dec_out, (-1, 16, dec_out.shape[1]))

        '''
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out * std + mean
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = dec_out[:, :self.pred_len, :]
        return dec_out

        # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, 5, dim=-1)
        return lags

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

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
