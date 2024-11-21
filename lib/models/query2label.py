# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict

from utils.util import *
from models.GATN_transformer import *

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
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

        # print("Size of input: ", input.size())
        # print("Size of weigth: ", self.weight.size())
        # print("Size of adj: ", adj.size())
        # print("Size of Support: ", support.size())

        output = torch.matmul(adj, support)

        # print("Output of GCN: ")
        # print(output)
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



class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = transfomer.d_model
        # print("hidden dim: ", hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # print("input proj: ", self.input_proj)

        self.query_embed = nn.Embedding(num_class, hidden_dim)
        # print("Query embedding: ", self.query_embed)

        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)



        ############################ 
        # GATN
        ############################
        self.pooling = nn.MaxPool2d(14, 14)

        
        
        # Topology
        if self.num_class == 20:
            s_adj = gen_A_correlation(self.num_class, 1.0, 0.4, "lib/models/topology/voc_correlation_adj.pkl")
            s_adj1 = gen_A_correlation(self.num_class, 0.4, 0.2, "lib/models/topology/voc_correlation_adj.pkl")
            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/embedding/voc_glove_word2vec.pkl", 'rb'))).float().to("cuda:0")

            
        elif self.num_class == 80:
            s_adj = gen_A_correlation(self.num_class, 1.0, 0.2, "lib/models/topology/coco_correlation_adj.pkl")
            s_adj1 = gen_A_correlation(self.num_class, 0.4, 0.2, "lib/models/topology/coco_correlation_adj.pkl")
            # s_adj = gen_A(num_classes, 0.4, 'data/coco/coco_adj.pkl')
            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/embedding/coco_glove_word2vec_80x300.pkl", 'rb'))).float().to("cuda:0")

        elif self.num_class ==11:
            s_adj = gen_A_correlation(self.num_class, 1.0, 0.2, "lib/models/topology/syntheticFiber_adj_11.pkl")
            s_adj1 = gen_A_correlation(self.num_class, 0.4, 0.2, "lib/models/topology/syntheticFiber_adj_11.pkl")

            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/topology/syntheticFiber_adj_11.pkl", 'rb'))).float().to("cuda:0")

        elif self.num_class ==12:
            s_adj = gen_A_correlation(self.num_class, 1.0, 0.2, "lib/models/topology/syntheticFiber_adj.pkl")
            s_adj1 = gen_A_correlation(self.num_class, 0.4, 0.2, "lib/models/topology/syntheticFiber_adj.pkl")

            self.s_Pos = torch.from_numpy(pickle.load(open("lib/models/embedding/syntheticFiberOneHot.pkl", 'rb'))).float().to("cuda:0")

        # Graph Convolutions
        self.gc1 = GraphConvolution(self.query_embed.weight.size(1), 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        
        s_adj = torch.from_numpy(s_adj).type(torch.FloatTensor)
        A_Tensor = s_adj.unsqueeze(-1)
        s_adj1 = torch.from_numpy(s_adj1).type(torch.FloatTensor)
        A_Tensor1 = s_adj1.unsqueeze(-1)


        self.transformerblock = GATNAttentionBlock(self.num_class)# TransformerBlock()

        # self.linear_A = nn.Linear(2 * self.num_class, self.num_class)
        self.linear_A = nn.Linear(2 * self.num_class, self.num_class)
        self.A_1 = A_Tensor.permute(2,0,1)
        self.A_2 = A_Tensor1.permute(2,0,1)
        self.A = A_Tensor.unsqueeze(0).permute(0,3,1,2)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]


    def forward(self, input):
        # ###################################################
        # # V1
        # ###################################################

        src, pos = self.backbone(input)
        # print("SRC: ", src)
        # print("SRC size: ", src.size())
        # print("SRC size: ", src.size())
        
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        # print("Input: ", input)
        # print("Input size: ", input.size())

        # print("SRC: ", src)
        # print("SRC size: ", src.size())

        # print("POS: ", pos)
        # print("pos size: ", pos.size())

        query_input = self.query_embed.weight

        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d

        print("hs size: ", hs.size())
        out = self.fc(hs[-1])
        # print("Out: ", out)
        print("Out size: ", out.size())
        # import ipdb; ipdb.set_trace()


        feature = self.pooling(src)
        feature = feature.view(feature.size(0), -1)
        
        inp = query_input

        # query_input

        # print("Inp: ", inp)
        # print("Inp size: ", inp.size())
        # print("query_input: ", query_input)
        # print("query_input size: ", query_input.size())
   
        # print("Inp: ", inp)
        # print("Inp size: ", inp.size())
        # print("Inp device: ", inp.device)
        # print("Inp type: ", inp.type)
        # print("Feature: ", feature)
        # print("Feature size: ", feature.size())
        # print("self.A_1",self.A_1.shape)
        # print("self.A_2",self.A_2.shape)
        adj, _ = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())

        # print("adj: ", adj)
        # print("adj size: ", adj.size())
        # print("adj device: ", adj.device)
        # print("adj type: ", adj.type)

        # print("Test 1 ")
        # adj = self.A_1.cuda()
        # print("adj_shape",adj.shape)
        
        adj = torch.squeeze(adj, 0) + torch.eye(self.num_class).type(torch.FloatTensor).cuda()
        adj = gen_adj(adj)

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = x.transpose(0, 1)

        # print("Output Model PRE matmul:")
        # # print(x)
        # print(x.size())
        # print(x.size(dim=1))
        # # print(x[:,0].size)

        x = torch.matmul(feature, x)

        # print("x", x)
        # print("size of x: ", x.size())

        # print("out", out)
        # print("size of out: ", out.size())

        
        inbetween = torch.cat((out, x), 1)
  
        # print("outpCombined", outpCombined)
        # print("size of outpCombined: ", outpCombined.size())

        # print("inbetween", inbetween)
        # print("size of inbetween: ", inbetween.size())

        # print("size of x: ", x.size())

        # print("size of out: ", out.size())

        outpCombined = self.linear_A(inbetween)

        # print("outpCombined", outpCombined)
        # print("size of outpCombined: ", outpCombined.size())

        return outpCombined
        # return out



        ###################################################
        # V2
        ###################################################

        # src, pos = self.backbone(input)
        # # print("SRC: ", src)
        # # print("SRC size: ", src.size())
        # # print("SRC size: ", src.size())
        
        # src, pos = src[-1], pos[-1]
        # # import ipdb; ipdb.set_trace()

        # # print("Input: ", input)
        # # print("Input size: ", input.size())

        # # print("SRC: ", src)
        # # print("SRC size: ", src.size())

        # # print("POS: ", pos)
        # # print("pos size: ", pos.size())

        # query_input = self.query_embed.weight
        # hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        # out = self.fc(hs[-1])
        # # import ipdb; ipdb.set_trace()


        # feature = self.pooling(src)
        # feature = feature.view(feature.size(0), -1)
        
        # # inp = input[0]
        # inp = self.s_Pos
   
        # # print("Inp: ", inp)
        # # print("Inp size: ", inp.size())
        # # print("Inp device: ", inp.device)
        # # print("Inp type: ", inp.type)
        # # print("Feature: ", feature)
        # # print("Feature size: ", feature.size())
        # # print("self.A_1",self.A_1.shape)
        # # print("self.A_2",self.A_2.shape)
        # adj, _ = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())

        # # print("adj: ", adj)
        # # print("adj size: ", adj.size())
        # # print("adj device: ", adj.device)
        # # print("adj type: ", adj.type)

        # # print("Test 1 ")
        # # adj = self.A_1.cuda()
        # # print("adj_shape",adj.shape)
        
        # adj = torch.squeeze(adj, 0) + torch.eye(self.num_class).type(torch.FloatTensor).cuda()
        # adj = gen_adj(adj)

        # # x = self.gc1(inp, adj)
        # # x = self.relu(x)
        # # x = self.gc2(x, adj)
        # # x = x.transpose(0, 1)

        # # print("Output Model PRE matmul:")
        # # # print(x)
        # # print(x.size())
        # # print(x.size(dim=1))
        # # # print(x[:,0].size)

        # # x = torch.matmul(feature, x)

        # # print("x", x)
        # # print("size of x: ", x.size())

        # # print("out", out)
        # # print("size of out: ", out.size())

        
        # # inbetween = torch.cat((out, x), 1)
        # outpCombined = torch.matmul(out, adj)
  
        # # print("outpCombined", outpCombined)
        # # print("size of outpCombined: ", outpCombined.size())

        # # print("inbetween", inbetween)
        # # print("size of inbetween: ", inbetween.size())

        # # print("size of x: ", x.size())

        # # print("size of out: ", out.size())

        # # outpCombined = self.linear_A(inbetween)

        # # print("outpCombined", outpCombined)
        # # print("size of outpCombined: ", outpCombined.size())

        # return outpCombined


        ###################################################
        # V3
        ###################################################

        # src, pos = self.backbone(input)
        # # print("SRC: ", src)
        # # print("SRC size: ", src.size())

        # # print("POS: ", pos)
        # # print("POS size: ", pos.size())

        
        # src, pos = src[-1], pos[-1]
        # # import ipdb; ipdb.set_trace()

        # # print("Input: ", input)
        # # print("Input size: ", input.size())

        # # print("SRC: ", src)
        # print("SRC size: ", src.size())

        # # print("POS: ", pos)
        # print("pos size: ", pos.size())

        # feature = self.pooling(src)
        # feature = feature.view(feature.size(0), -1)
        
        # inp = self.s_Pos
   
        # # print("Inp: ", inp)
        # # print("Inp size: ", inp.size())
        # # print("Inp device: ", inp.device)
        # # print("Inp type: ", inp.type)
        # # print("Feature: ", feature)
        # # print("Feature size: ", feature.size())
        # # print("self.A_1",self.A_1.shape)
        # # print("self.A_2",self.A_2.shape)
        # adj, _ = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())

        # # print("adj: ", adj)
        # # print("adj size: ", adj.size())
        # # print("adj device: ", adj.device)
        # # print("adj type: ", adj.type)

        # # print("Test 1 ")
        # # adj = self.A_1.cuda()
        # # print("adj_shape",adj.shape)
        
        # adj = torch.squeeze(adj, 0) + torch.eye(self.num_class).type(torch.FloatTensor).cuda()
        # adj = gen_adj(adj)

        # # print("adj: ", adj)
        # print("adj size: ", adj.size())
        

       
        # # import ipdb; ipdb.set_trace()

        # x = self.gc1(inp, adj)
        # x = self.relu(x)
        # x = self.gc2(x, adj)
        # print("size of x: ", x.size())

        # x = x.transpose(0, 1)
        # print("size of x: ", x.size())

        # # print("Output Model PRE matmul:")
        # # # print(x)
        # # print(x.size())
        # # print(x.size(dim=1))
        # # # print(x[:,0].size)

        # x = torch.matmul(feature, x)

        # # print("x", x)
        # print("size of x: ", x.size())

        # query_input = self.query_embed.weight
        # hs = self.transformer(self.input_proj(src), query_input, adj)[0] # B,K,d
        # out = self.fc(hs[-1])

        # # print("out", out)
        # # print("size of out: ", out.size())

        
        # inbetween = torch.cat((out, x), 1)
  
        # # print("outpCombined", outpCombined)
        # # print("size of outpCombined: ", outpCombined.size())

        # # print("inbetween", inbetween)
        # # print("size of inbetween: ", inbetween.size())

        # # print("size of x: ", x.size())

        # # print("size of out: ", out.size())

        # outpCombined = self.linear_A(inbetween)

        # # print("outpCombined", outpCombined)
        # # print("size of outpCombined: ", outpCombined.size())

        # return outpCombined
        # # return out



        ###################################################
        # V4
        ###################################################

        # src, pos = self.backbone(input)
        # # print("SRC: ", src)
        # # print("SRC size: ", src.size())
        # # print("SRC size: ", src.size())
        
        # src, pos = src[-1], pos[-1]
        # # import ipdb; ipdb.set_trace()

        # # print("Input: ", input)
        # # print("Input size: ", input.size())

        # # print("SRC: ", src)
        # # print("SRC size: ", src.size())

        # # print("POS: ", pos)
        # # print("pos size: ", pos.size())

        # # feature = self.pooling(src)
        # # feature = feature.view(feature.size(0), -1)
        
        # # inp = self.s_Pos
   
        # # print("Inp: ", inp)
        # # print("Inp size: ", inp.size())
        # # print("Inp device: ", inp.device)
        # # print("Inp type: ", inp.type)
        # # print("Feature: ", feature)
        # # print("Feature size: ", feature.size())
        # # print("self.A_1",self.A_1.shape)
        # # print("self.A_2",self.A_2.shape)
        # adj, _ = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())

        # # print("adj: ", adj)
        # # print("adj size: ", adj.size())
        # # print("adj device: ", adj.device)
        # # print("adj type: ", adj.type)

        # # print("Test 1 ")
        # # adj = self.A_1.cuda()
        # # print("adj_shape",adj.shape)
        
        # adj = torch.squeeze(adj, 0) + torch.eye(self.num_class).type(torch.FloatTensor).cuda()
        # adj = gen_adj(adj)

        # query_input = self.query_embed.weight
        # hs = self.transformer(self.input_proj(src), query_input, adj)[0] # B,K,d
        # out = self.fc(hs[-1])
        # # import ipdb; ipdb.set_trace()

        # # x = self.gc1(inp, adj)
        # # x = self.relu(x)
        # # x = self.gc2(x, adj)
        # # x = x.transpose(0, 1)

        # # print("Output Model PRE matmul:")
        # # # print(x)
        # # print(x.size())
        # # print(x.size(dim=1))
        # # # print(x[:,0].size)

        # # x = torch.matmul(feature, x)

        # # print("x", x)
        # # print("size of x: ", x.size())

        # # print("out", out)
        # # print("size of out: ", out.size())

        
        # # inbetween = torch.cat((out, x), 1)
        # outpCombined = torch.matmul(out, adj)

        # # print("outpCombined", outpCombined)
        # # print("size of outpCombined: ", outpCombined.size())

        # # print("inbetween", inbetween)
        # # print("size of inbetween: ", inbetween.size())

        # # print("size of x: ", x.size())

        # # print("size of out: ", out.size())

        # # outpCombined = self.linear_A(inbetween)

        # # print("outpCombined", outpCombined)
        # # print("size of outpCombined: ", outpCombined.size())

        # return outpCombined
        # # return out





    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Qeruy2Label(
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_class
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    

    return model
        
        



# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# class GATNResnet(nn.Module):
#     def __init__(self, model_name, num_classes, in_channel=300, t1=0.0, adj_file=None):
#         super(GATNResnet, self).__init__()

#         # Create multiple backbones
#         model = load_model(model_name)
#         self.backbone = nn.Sequential(
#         model.conv1,
#         model.bn1,
#         model.relu,
#         model.maxpool,
#         model.layer1,
#         model.layer2,
#         model.layer3,
#         model.layer4,
#         )
#         self.num_classes = num_classes
#         self.pooling = nn.MaxPool2d(14, 14)

#         # Graph Convolutions
#         self.gc1 = GraphConvolution(in_channel, 1024)
#         self.gc2 = GraphConvolution(1024, 2048)
#         self.relu = nn.LeakyReLU(0.2)
        
#         # Topology
#         if num_classes == 20:
#             s_adj = gen_A_correlation(num_classes, 1.0, 0.4, "./model/topology/voc_correlation_adj.pkl")
#             s_adj1 = gen_A_correlation(num_classes, 0.4, 0.2, "./model/topology/voc_correlation_adj.pkl")
#         elif num_classes == 80:
#             s_adj = gen_A_correlation(num_classes, 1.0, 0.2, "./model/topology/coco_correlation_adj.pkl")
#             s_adj1 = gen_A_correlation(num_classes, 0.4, 0.2, "./model/topology/coco_correlation_adj.pkl")
#             # s_adj = gen_A(num_classes, 0.4, 'data/coco/coco_adj.pkl')
#         elif num_classes ==12:
#             s_adj = gen_A_correlation(num_classes, 1.0, 0.2, "./model/topology/syntheticFiber_adj.pkl")
#             s_adj1 = gen_A_correlation(num_classes, 0.4, 0.2, "./model/topology/syntheticFiber_adj.pkl")

#         s_adj = torch.from_numpy(s_adj).type(torch.FloatTensor)
#         A_Tensor = s_adj.unsqueeze(-1)
#         s_adj1 = torch.from_numpy(s_adj1).type(torch.FloatTensor)
#         A_Tensor1 = s_adj1.unsqueeze(-1)


#         self.transformerblock = AttentionBlock(num_classes)# TransformerBlock()

#         self.linear_A = nn.Linear(80, 80)
#         self.A_1 = A_Tensor.permute(2,0,1)
#         self.A_2 = A_Tensor1.permute(2,0,1)
#         self.A = A_Tensor.unsqueeze(0).permute(0,3,1,2)

#         # image normalization
#         self.image_normalization_mean = [0.485, 0.456, 0.406]
#         self.image_normalization_std = [0.229, 0.224, 0.225]

#     def forward(self, feature, inp, label = None, iters = 1, max_iters = 256600):
#         batch_size = feature.shape[0]

#         # print("Feature: ", feature)
#         # print("Feature size: ", feature.size())


#         feature = self.backbone(feature)
#         # print("Feature size: ", feature.size())
#         feature = self.pooling(feature)
#         feature = feature.view(feature.size(0), -1)
        
#         inp = inp[0]

#         # print("Inp: ", inp)
#         # print("Inp size: ", inp.size())
#         # print("Feature: ", feature)
#         # print("Feature size: ", feature.size())
#         # print("self.A_1",self.A_1.shape)
#         # print("self.A_2",self.A_2.shape)
#         adj, _ = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())

#         # print("Test 1 ")
#         # adj = self.A_1.cuda()
#         # print("adj_shape",adj.shape)
        
#         adj = torch.squeeze(adj, 0) + torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
#         adj = gen_adj(adj)

#         x = self.gc1(inp, adj)
#         x = self.relu(x)
#         x = self.gc2(x, adj)
#         x = x.transpose(0, 1)

#         # print("Output Model PRE matmul:")
#         # # print(x)
#         # print(x.size())
#         # print(x.size(dim=1))
#         # # print(x[:,0].size)

#         x = torch.matmul(feature, x)

#         # print("Output Model POST matmul:")
#         # # print(x)
#         # print(x.size())
#         # print(x.size(dim=1))
#         # #print(x[:,0].size())

#         return x

#     def get_config_optim(self, lr, lrp, lrt):
#         config_optim = []
#         config_optim.append({'params': self.backbone.parameters(), 'lr': lr * lrp})
#         config_optim.append({'params': self.transformerblock.parameters(), 'lr': lr * lrt})
#         config_optim.append({'params': self.gc1.parameters(), 'lr': lr})
#         config_optim.append({'params': self.gc2.parameters(), 'lr': lr})
#         return config_optim



# def gatn_resnet(num_classes, t1, pretrained=True, adj_file=None, in_channel=300):
#     # return GATNResnet('resnext101_32x16d_swsl', num_classes, t1=t1, adj_file=adj_file, in_channel=in_channel)
#     # return GATNResnet('resnext101_32x8d_swsl', num_classes, t1=t1, adj_file=adj_file, in_channel=in_channel)

#     return GATNResnet('resnext101_32x8d_ssl', num_classes, t1=t1, adj_file=adj_file, in_channel=in_channel)
    
#     # return GATNResnet('resnet101', num_classes, t1=t1, adj_file=adj_file, in_channel=in_channel)