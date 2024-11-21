# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, BCELoss, Dropout, Softmax, Sigmoid, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class GATNAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GATNAttention, self).__init__()
        self.vis = False
        if hidden_size == 11 or hidden_size == 13 or hidden_size == 17:
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = 4 
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads) # hidden size = 1280, hidden size = embedding dimension?  
        self.all_head_size = self.num_attention_heads * self.attention_head_size # theoritical ,it should be equal to hidden_size
        # self.all_head_size = self.hidden_size # theoritical ,it should be equal to hidden_size
        # print("All head size: ", self.all_head_size)
        
        # print("num_attention_heads", self.num_attention_heads)
        # print("attention_head_size", self.attention_head_size)
        # print("All head size: ",  self.all_head_size)
        # print("hidden size: ",  self.hidden_size)

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # print("Transpose x: ", x)
        # print("Size of Transpose x: ", x.size())
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #(8 257) + (16 80)
        # print("New x shape: ", new_x_shape)
        x = x.view(*new_x_shape) # reshape
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print("Hidden states: ", hidden_states.size)
        # print("Size of Hidden states: ", hidden_states.size())
        mixed_query_layer = self.query(hidden_states) # 1 80 80
        mixed_key_layer = self.key(hidden_states) # 1 80 80
        mixed_value_layer = self.value(hidden_states) # 1 80 80

        # print("Size of mixed_query_layer: ", mixed_query_layer.size())
        # print("Size of mixed_key_layer: ", mixed_key_layer.size())
        # print("Size of mixed_value_layer: ", mixed_value_layer.size())
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Q X KT
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # /sqrt(d) norm
        attention_probs = self.softmax(attention_scores) # softmax
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # W X V
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class GATNAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(GATNAttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm1 = LayerNorm(self.hidden_size, eps=1e-6)
        # self.attention_norm2 = LayerNorm(self.hidden_size, eps=1e-6)
        self.attn1 = GATNAttention(hidden_size)
        self.attn2 = GATNAttention(hidden_size)
        # print("AttentionBlock test")
        # print("self.attn1",self.attn1)
        # print("self.attn2",self.attn2)

        # self.attn3 = Attention(hidden_size)
        # self.attn4 = Attention(hidden_size)

    def forward(self, x, y = None):
        # print("GATN x: ", x)
        # print("GATN x size: ", x.size())
        x_save = x
        x1 = self.attention_norm1(x)
        x1, weights = self.attn1(x1)

        # if y != None:
        #     y_save = y
        #     y1 = self.attention_norm1(y)
        #     y1, weights = self.attn3(y1)
        #     x1 = x1 + y1

        x2 = self.attention_norm1(x_save)
        # x2 = self.attention_norm2(x_save)
        x2, weights = self.attn2(x2)
        
        # if y!= None:
        #     y2 = self.attention_norm1(y_save)
        #     y2, weights = self.attn4(y2)
        #     x2 = x2 + y2

        x = torch.bmm(x1, x2)
        return x, weights
