import copy
import pickle
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention
import torch.distributed as dist
from models.model_GAT import *
from utils.util import *
from models.backbone import build_backbone


from utils.misc import clean_state_dict

class GroupWiseLinear(nn.Module):

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
    

class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)

class LLCDTransformer(nn.Module):

    def __init__(self, backbone, args,
                 activation="relu", 
                 normalize_before=False,
                 return_intermediate_dec=False, 
                 rm_self_attn_dec=True, 
                 rm_first_self_attn=True,
                 ):
        
        super().__init__()

        self.backbone = backbone

        embed_len_decoder = 100 if args.dec_layers < 0 else args.dec_layers
        if embed_len_decoder > args.num_class:
            embed_len_decoder = args.num_class

        # switching to 768 initial embeddings
        self.decoder_embedding = args.hidden_dim

        self.embed_standart = nn.Linear(2048, self.decoder_embedding)

        self.query_embed = nn.Embedding(args.num_class, self.decoder_embedding)
        self.query_embed.requires_grad_(False)
        
        self.num_classes = args.num_class
        self.dmod = args.hidden_dim

        # Graph Encoders:
        self.graphEncoder = GATN(args.num_class, args.hidden_dim)


        self.group_fc = GroupFC(args.num_class)

        self.duplicate_pooling = torch.nn.Parameter(torch.Tensor(args.num_class, args.hidden_dim, 1))

        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))

        self.CropLevels = args.CropLevels
        self.imgSize = args.img_size


        # LLCD decoder:
        decoder_layer = LLCD_Layer(args.hidden_dim, args.nheads, args.dim_feedforward,
                                                        args.dropout, activation, normalize_before, args.num_class)
        decoder_norm = nn.LayerNorm(args.hidden_dim)
        self.decoder = LLCD_Decoder(decoder_layer, args.dec_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, d_model=args.hidden_dim, dim_feedforward=args.dim_feedforward, dropout=args.dropout, numclasses=args.num_class)
        

    def crop_forward(self, input, CropLevels):
        bs, c, h, w = input.shape
        if CropLevels != 0:
            cropDist = w // CropLevels
        else:
            cropDist = w

        all_features = []
        all_pos = []

        for j in range(1, CropLevels+1):

            for k in range(1, CropLevels+1): 
                cropped_img = input[:, :, cropDist * (k-1):0 + cropDist * k, cropDist * (j-1):0 + cropDist * j]
                resized_img = F.interpolate(cropped_img, size=(self.imgSize, self.imgSize), mode='bilinear', align_corners=False)
                src, pos = self.backbone(resized_img)
                src, pos = src[-1], pos[-1]
                src = src.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
                pos = pos.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
                all_features.append(src)
                all_pos.append(pos)

        # Resize the input to SImg * 2^i x SImg * 2^i
        resized_img = F.interpolate(input, size=(self.imgSize, self.imgSize), mode='bilinear', align_corners=False)

        src, pos = self.backbone(resized_img)
        src, pos = src[-1], pos[-1]

        src = src.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        pos = pos.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        all_features.append(src)
        all_pos.append(pos)
 
        # Combine features from all levels (simple concatenation here, could be weighted)
        combined_features = torch.cat(all_features, dim=0)  # Combine along the token dimension
        combined_pos = torch.cat(all_pos, dim=0)

        return combined_features,  combined_pos
    

    def forward(self, input):
        bs, c, h, w = input.shape

        imgsrc, imgpos = self.crop_forward(input, self.CropLevels )

        imgsrc = self.embed_standart(imgsrc)

        query_embedding = self.query_embed.weight

        query_tgt = query_embedding.unsqueeze(1).expand(-1, bs, -1)

        adjMat = self.graphEncoder(query_embedding)

        adjMat = adjMat.unsqueeze(0).repeat(bs, 1, 1) + torch.eye(self.dmod).type(torch.FloatTensor).cuda()

        adjMat = adjMat.view(self.dmod, bs, self.dmod)

        h = self.decoder(query_tgt, imgsrc, adjMat, query_pos = query_tgt, pos=imgpos, inpimage=input)

        h = h.transpose(0, 1)
        out_extrap = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
        self.group_fc(h, self.duplicate_pooling, out_extrap)

        h_out = out_extrap.flatten(1)

        h_out += self.duplicate_pooling_bias
        logits = h_out
        
        return(logits)

class TrimodalTransformer_MOD(nn.Module):

    def __init__(self, args,
                 activation="relu", 
                 normalize_before=False,
                 return_intermediate_dec=False, 
                 rm_self_attn_dec=True, 
                 rm_first_self_attn=True,
                 ):
        
        super().__init__()

        embed_len_decoder = 100 if args.dec_layers < 0 else args.dec_layers
        if embed_len_decoder > args.num_class:
            embed_len_decoder = args.num_class

        self.decoder_embedding = args.hidden_dim

        self.embed_standart = nn.Linear(2048, self.decoder_embedding)

        self.query_embed = nn.Embedding(args.num_class, self.decoder_embedding)
        self.query_embed.requires_grad_(False)
        
        self.num_classes = args.num_class
        self.dmod = args.hidden_dim

        # Graph Encoders:
        self.graphEncoder = GATN(args.num_class, args.hidden_dim)


        self.group_fc = GroupFC(args.num_class)

        self.duplicate_pooling = torch.nn.Parameter(torch.Tensor(args.num_class, args.hidden_dim, 1))

        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))

        # Trimodal decoder:
        decoder_layer = TrimodalTransformerDecoderLayer(args.hidden_dim, args.nheads, args.dim_feedforward,
                                                        args.dropout, activation, normalize_before, args.num_class)
        decoder_norm = nn.LayerNorm(args.hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, args.dec_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, d_model=args.hidden_dim, dim_feedforward=args.dim_feedforward, dropout=args.dropout, numclasses=args.num_class)
    
    def forward(self, input):
        bs, c, h, w = input.shape

        imgsrc = input.flatten(2).permute(2, 0, 1) 

        imgsrc = self.embed_standart(imgsrc)

        query_embedding = self.query_embed.weight

        query_tgt = query_embedding.unsqueeze(1).expand(-1, bs, -1)

        adjMat = self.graphEncoder(query_embedding)

        # Adjust adjMat to have a batch dimension
        adjMat = adjMat.unsqueeze(0).repeat(bs, 1, 1) + torch.eye(self.dmod).type(torch.FloatTensor).cuda()

        adjMat = adjMat.view(self.dmod, bs, self.dmod)

        h = self.decoder(query_tgt, imgsrc, adjMat, query_pos = query_tgt)

        h = h.transpose(0, 1)
        out_extrap = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
        self.group_fc(h, self.duplicate_pooling, out_extrap)

        h_out = out_extrap.flatten(1)

        h_out += self.duplicate_pooling_bias
        logits = h_out

        

        return(logits)
    
    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(0))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))

class LLCD_Layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_classes=11):
        super().__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_image = MultiheadAttention(d_model, nhead, dropout=dropout)  # For image features
        self.multihead_attn_graph = MultiheadAttention(d_model, nhead, dropout=dropout)  # For graph features

        self.cross_attn_graph_to_img = MultiheadAttention(d_model, nhead, dropout=dropout)  # Graph attends to image

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout) 
        self.dropout5 = nn.Dropout(dropout)  
        self.dropout6 = nn.Dropout(dropout) 
        
        self.normalize_before = normalize_before

        # MLP layers for refining cross-attention outputs
        self.mlp_img_to_img = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model)
        )
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_image, memory_graph,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        
        

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt12 = self.multihead_attn(tgt, memory_image, memory_image)[0]
        tgt = tgt + self.dropout2(tgt12)
        tgt = self.norm2(tgt)


        tgt2, _ = self.multihead_attn_image(query=memory_image,
                                         key=memory_graph,
                                         value=memory_graph, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt2 = memory_image + self.dropout4(tgt2)
        tgt2 = self.norm4(tgt2)

        


        tgt3 = self.multihead_attn_graph(query=memory_graph, 
                                         key=memory_image,
                                         value=memory_image, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)[0]
        tgt3 = memory_graph + self.dropout5(tgt3)
        tgt3 = self.norm5(tgt3)

        
        tgt4, weight = self.cross_attn_graph_to_img(query=tgt2,
                                         key=tgt3,
                                         value=tgt3, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt4 = tgt2 + self.dropout6(tgt4)
        tgt4 = self.norm6(tgt4)

        tgt12 = self.mlp_img_to_img(tgt)
        tgt = tgt + self.dropout3(tgt12)
        tgt = self.norm3(tgt)


        return tgt , tgt4, tgt3, weight


    def forward(self, tgt, memory_image, memory_graph,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory_image, memory_graph, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        





class LLCD_Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=2048, dim_feedforward=8192, dropout=0.1, activation='relu', numclasses=80):
        super().__init__()

        self.layers = nn.ModuleList([
            decoder_layer
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers


        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )


    def forward(self, tgt: Tensor, memory_image: Tensor, memory_graph: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, 
                inpimage : Optional[Tensor] = None):

        output = tgt
        for layer in self.layers:
            output, memory_image, memory_graph, weigth = layer(
                output, 
                memory_image,
                memory_graph,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

            i += 1
        return output


def buildLLCDTransformer(args):

    model_params = {'args': args, 'num_classes': args.num_class}

    if args.backbone == "TResnetL_V2" or args.backbone == "CvT_w24" or args.backbone == "resnet101": 
        backbone = build_backbone(args)

    elif args.backbone == "tresnet_l":
        backbone = TResnetL(model_params)
        
    model = LLCDTransformer(
        backbone=backbone,
        args=args,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
        rm_self_attn_dec=not args.keep_other_self_attn_dec, 
        rm_first_self_attn=not args.keep_first_self_attn_dec,
    )
    
    return model

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


