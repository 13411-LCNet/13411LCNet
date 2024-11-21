# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import copy
import logging
import math

from os.path import join as pjoin
import pickle

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from torch.nn import CrossEntropyLoss, BCELoss, Dropout, Softmax, Sigmoid, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from lib.utils.util import *

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #(8 257) + (16 80)
        x = x.view(*new_x_shape) # reshape
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # 1 80 80
        mixed_key_layer = self.key(hidden_states) # 1 80 80
        mixed_value_layer = self.value(hidden_states) # 1 80 80
        
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

        self.attn3 = GATNAttention(hidden_size)
        self.attn4 = GATNAttention(hidden_size)

    def forward(self, x, y = None):

        x_save = x
        x1 = self.attention_norm1(x)
        x1, weights = self.attn1(x1)

        x2 = self.attention_norm1(x_save)
        # x2 = self.attention_norm2(x_save)
        x2, weights = self.attn2(x2)

        x = torch.bmm(x1, x2)
        return x, weights


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def write_matrix(matrix_a, files):
    matrix = matrix_a.cpu()
    matrix = matrix.detach().numpy()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            files.write(str(matrix[i][j]) + ",")
        files.write("\n")


class GATN(nn.Module):
    def __init__(self, num_classes, in_channel=300, t1=0.0, adj_file=None,):
        super().__init__()

        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        
        
       # Topology
        if self.num_classes == 20:
            s_adj = gen_A_correlation(self.num_classes, 1.0, 0.4, "lib/models/topology/voc_correlation_adj.pkl")
            s_adj1 = gen_A_correlation(self.num_classes, 0.4, 0.2, "lib/models/topology/voc_correlation_adj.pkl")
            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/embedding/voc_glove_word2vec.pkl", 'rb'))).float().to("cuda:0")

            
        elif self.num_classes == 80:
            s_adj = gen_A_correlation(self.num_classes, 1.0, 0.2, "lib/models/topology/coco_correlation_adj.pkl")
            s_adj1 = gen_A_correlation(self.num_classes, 0.4, 0.2, "lib/models/topology/coco_correlation_adj.pkl")
            # s_adj = gen_A(num_classes, 0.4, 'data/coco/coco_adj.pkl')
            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/embedding/coco_glove_word2vec_80x300.pkl", 'rb'))).float().to("cuda:0")

        elif self.num_classes ==11:
            s_adj = gen_A_correlation(self.num_classes, 1.0, 0.2, "lib/models/topology/syntheticFiber_adj_11.pkl")
            s_adj1 = gen_A_correlation(self.num_classes, 0.4, 0.2, "lib/models/topology/syntheticFiber_adj_11.pkl")

            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/topology/syntheticFiber_adj_11.pkl", 'rb'))).float().to("cuda:0")

        elif self.num_classes ==12:
            s_adj = gen_A_correlation(self.num_classes, 1.0, 0.2, "lib/models/topology/syntheticFiber_adj.pkl")
            s_adj1 = gen_A_correlation(self.num_classes, 0.4, 0.2, "lib/models/topology/syntheticFiber_adj.pkl")

            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/embedding/syntheticFiberOneHot.pkl", 'rb'))).float().to("cuda:0")


        # Graph Convolutions
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, in_channel)
        self.relu = nn.LeakyReLU(0.2)


        s_adj = torch.from_numpy(s_adj).type(torch.FloatTensor)
        A_Tensor = s_adj.unsqueeze(-1)
        s_adj1 = torch.from_numpy(s_adj1).type(torch.FloatTensor)
        A_Tensor1 = s_adj1.unsqueeze(-1)


        self.transformerblock = GATNAttentionBlock(num_classes)# TransformerBlock()

        self.linear_A = nn.Linear(80, 80)
        self.A_1 = A_Tensor.permute(2,0,1)
        self.A_2 = A_Tensor1.permute(2,0,1)
        self.A = A_Tensor.unsqueeze(0).permute(0,3,1,2)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, inp, label = None, iters = 1, max_iters = 256600):

        adj, wgt = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())
        
        adj = torch.squeeze(adj, 0) + torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = gen_adj(adj)


        x = self.gc1(inp, adj)

        x = self.relu(x)

        x = self.gc2(x, adj)

        x = torch.matmul( x.transpose(0, 1), x)


        # print("===========================================")
        # print("2 adjMat shape: ", adj.shape)
        # print("2 adjMat: ", adj)
        # print("===========================================")

        # export_graph_matplotlib(adj)

        return x


def export_graph_matplotlib(adj_matrix, output_path="graph_visualization.png"):
    """
    Exports a graph visualization from an adjacency matrix using Matplotlib.
    
    Args:
    - adj_matrix: Adjacency matrix (torch.Tensor or numpy.ndarray).
    - output_path: Path to save the graph visualization.
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()  # Convert to numpy.

    # Build the graph
    graph = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0.01:  # Example threshold
                if i != j:
                    graph.add_edge(i, j, weight=adj_matrix[i, j])

    # Draw the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph, iterations=100, seed=173)  # Position nodes using the spring layout algorithm
    # pos = nx.shell_layout(graph)  # Position nodes using the spring layout algorithm


    # Extract edge weights for line width
    edges = graph.edges(data=True)
    weights = [data['weight'] for _, _, data in edges]

    # Normalize weights for visualization (optional)
    max_weight = max(weights) if weights else 1
    linewidths = [3 * (w / max_weight) for w in weights]  # Scale thickness


    nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, font_size=10, width=linewidths)
    
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))

    # Save the graph visualization
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Graph visualization saved to {output_path}")