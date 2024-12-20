import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, pos):
        gp_embds = Variable(self.pe[pos[:, :, 0]], requires_grad=False)
        lp_embds = Variable(self.pe[pos[:, :, 1]], requires_grad=False)
        pp_embds = Variable(self.pe[pos[:, :, 2]], requires_grad=False)
        return gp_embds, lp_embds, pp_embds

class PositionLayer(nn.Module):
    def __init__(self, p_embd=None, p_embd_dim=16, zero_weight=False):
        super(PositionLayer, self).__init__()
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim

        if zero_weight:
            self.pWeight = nn.Parameter(torch.zeros(3))
        else:
            self.pWeight = nn.Parameter(torch.ones(3))
        
        if p_embd == 'embd':
            self.g_embeddings = nn.Embedding(41, p_embd_dim)
            self.l_embeddings = nn.Embedding(21, p_embd_dim)
            self.p_embeddings = nn.Embedding(11, p_embd_dim)
        elif p_embd == 'embd_a':
            self.g_embeddings = nn.Embedding(100, p_embd_dim)
            self.l_embeddings = nn.Embedding(50, p_embd_dim)
            self.p_embeddings = nn.Embedding(30, p_embd_dim)
            self.gp_Linear = nn.Linear(p_embd_dim, 1)
            self.lp_Linear = nn.Linear(p_embd_dim, 1)
            self.pp_Linear = nn.Linear(p_embd_dim, 1)
        elif p_embd == 'embd_b':
            self.g_embeddings = nn.Embedding(41, p_embd_dim)
            self.l_embeddings = nn.Embedding(21, p_embd_dim)
            self.p_embeddings = nn.Embedding(11, p_embd_dim)
        elif p_embd == 'embd_c':
            self.pe = PositionalEncoding(p_embd_dim, 100)

    def forward(self, sentpres, pos):
        # sentpres: (batch_n, doc_l, output_dim*2)
        if self.p_embd in utils.embd_name:
            pos = pos[:, :, 3:6].long()
        else:
            pos = pos[:, :, :3]
        if self.p_embd == 'embd':
            gp_embds = torch.tanh(self.g_embeddings(pos[:, :, 0]))
            lp_embds = torch.tanh(self.l_embeddings(pos[:, :, 1]))
            pp_embds = torch.tanh(self.p_embeddings(pos[:, :, 2]))
            sentpres = torch.cat((sentpres, gp_embds, lp_embds, pp_embds), dim=2)
        elif self.p_embd == 'embd_a':
            gp_embds = self.g_embeddings(pos[:, :, 0])
            lp_embds = self.l_embeddings(pos[:, :, 1])
            pp_embds = self.p_embeddings(pos[:, :, 2])
            sentpres = sentpres + self.pWeight[0] * torch.tanh(self.gp_Linear(gp_embds)) + \
                                  self.pWeight[1] * torch.tanh(self.lp_Linear(lp_embds)) + \
                                  self.pWeight[2] * torch.tanh(self.pp_Linear(pp_embds))
        elif self.p_embd == 'embd_b':
            gp_embds = self.g_embeddings(pos[:, :, 0])
            lp_embds = self.l_embeddings(pos[:, :, 1])
            pp_embds = self.p_embeddings(pos[:, :, 2])
            sentpres = sentpres + self.pWeight[0] * torch.tanh(gp_embds) + \
                                  self.pWeight[1] * torch.tanh(lp_embds) + \
                                  self.pWeight[2] * torch.tanh(pp_embds)
        elif self.p_embd == 'embd_c':
            gp_embds, lp_embds, pp_embds = self.pe(pos)
            sentpres = sentpres + self.pWeight[0] * gp_embds + \
                                  self.pWeight[1] * lp_embds + \
                                  self.pWeight[2] * pp_embds                   
        elif self.p_embd == 'cat':
            sentpres = torch.cat((sentpres, pos), dim=2)
        elif self.p_embd =='add':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
        elif self.p_embd =='add1':
            sentpres = sentpres + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
        elif self.p_embd =='add2':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[2] * pos[:, :, 2:3]
        elif self.p_embd =='add3':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2]
        elif self.p_embd =='addg':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1]
        elif self.p_embd =='addl':
            sentpres = sentpres + self.pWeight[1] * pos[:, :, 1:2]
        elif self.p_embd =='addp':
            sentpres = sentpres + self.pWeight[2] * pos[:, :, 2:3]
            
        return sentpres
        
    def init_embedding(self):
        gp_em_w = [[i/40] * self.p_embd_dim for i in range(41)]
        self.g_embeddings.weight = nn.Parameter(torch.FloatTensor(gp_em_w))
        lp_em_w = [[i/20] * self.p_embd_dim for i in range(21)]
        self.l_embeddings.weight = nn.Parameter(torch.FloatTensor(lp_em_w))
        pp_em_w = [[i/10] * self.p_embd_dim for i in range(11)]
        self.p_embeddings.weight = nn.Parameter(torch.FloatTensor(pp_em_w))
        
         
class InterSentenceSPPLayer(nn.Module):
    def __init__(self, hidden_dim, num_levels=4, pool_type='max_pool'):
        super(InterSentenceSPPLayer, self).__init__()
        self.linearK = nn.Linear(hidden_dim, hidden_dim)
        self.linearQ = nn.Linear(hidden_dim, hidden_dim)
        self.num_levels = num_levels
        self.pool_type = pool_type
        if self.pool_type == 'max_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        else:
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)])
        
    def forward(self, sentpres, is_softmax=False):
        # sentpres: (batch_n, doc_l, output_dim*2)
        doc_l = sentpres.size(1)
        key = self.linearK(sentpres)
        query = self.linearQ(sentpres)
        d_k = query.size(-1)
        features = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # features: (batch_n, doc_l, doc_l)
        if is_softmax:
            features = F.softmax(features, dim=2)
            # print(torch.sum(features, dim=2))

        features = torch.tanh(features)
        self.ft =  features
        pooling_layers = []
        for pooling in self.SPP:
            tensor = pooling(features)
            pooling_layers.append(tensor)
            
        # print([x.size() for x in pooling_layers])
        self.features = torch.cat(pooling_layers, dim=-1)
        return self.features  
        
        
class InterSentenceSPPLayer3(nn.Module):
    def __init__(self, hidden_dim, num_levels=4, pool_type='max_pool', active_func='tanh'):
        super(InterSentenceSPPLayer3, self).__init__()
        self.linearK = nn.Linear(hidden_dim, hidden_dim)
        self.linearQ = nn.Linear(hidden_dim, hidden_dim)
        self.num_levels = num_levels
        self.pool_type = pool_type
        if self.pool_type == 'max_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        elif self.pool_type == 'avg_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)])
        else:
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)] + [nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        if active_func == 'tanh':
            self.active_func = nn.Tanh()
        elif active_func == 'relu':
            self.active_func = nn.ReLU()
        elif active_func == 'softmax':
            self.active_func = nn.Softmax(dim=2)
        else:
            self.active_func = None

    def forward(self, sentpres):
        # sentpres: (batch_n, doc_l, output_dim*2)
        doc_l = sentpres.size(1)
        key = self.linearK(sentpres)
        query = self.linearQ(sentpres)
        d_k = query.size(-1)
        features = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # features: (batch_n, doc_l, doc_l)

        if self.active_func is not None:
            features = self.active_func(features)
        self.ft =  features
        pooling_layers = []
        for pooling in self.SPP:
            tensor = pooling(features)
            pooling_layers.append(tensor)
            
        # print([x.size() for x in pooling_layers])
        self.features = torch.cat(pooling_layers, dim=-1)
        return self.features

def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings


# %%

def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了



    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k


# %%

def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]

    att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
    att_logits /= math.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e9)  # mask掉为0的部分，设为无穷大

    att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)

    if dropout is not None:
        att_scores = dropout(att_scores)

    # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
    return torch.matmul(att_scores, v), att_scores

