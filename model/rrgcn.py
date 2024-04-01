import math

from numpy.testing._private.utils import requires_memory
from dgl.convert import graph
from dgl.nn import GraphConv,TAGConv
from scipy.sparse import dia

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from model.layers import UnionRGCNLayer, CompGCNCovLayer
from model.hrgcn import HawkesRGCNLayer
from src.utils import merge_graphs
#from src.model import BaseRGCN
from model.decoder import ConvTransE, ConvTransR, InteractE
from model.eventmodel import EventDecoderE, EventDecoderR
from model.focalloss import FocalLoss

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        #print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == 'compgcn':
            return CompGCNCovLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=act, dropout=self.dropout, opn=self.opn, rel_emb=self.rel_emb)
        elif self.encoder_name == 'hrgcn':
            return HawkesRGCNLayer(self.h_dim, self.h_dim, self.num_rels, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "hrgcn":
            node_id = g.ndata['id'].squeeze()
            edge_id = g.edata['type'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r)
            return g.ndata.pop('h'), []
        if self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            edge_id = g.edata['type'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            g.edata['h'] = init_rel_emb[edge_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                x, r = layer(g, x, r)
            return x, r
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, 
                 theta=1, entity_prediction=False, relation_prediction=False, raw_input=False, use_cuda=False,
                 gpu = 0):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.graph_h_dim = h_dim*2
        self.layer_norm = layer_norm
        self.h = None
        self.h_0 = None
        self.graph_h = None
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.theta = theta
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.raw_input = raw_input
        self.gpu = gpu

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()      #对所有的关系做嵌入
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()      #对实体做嵌入
        torch.nn.init.normal_(self.emb_ent)

        #注意力中的网络层
        # self.w_q = torch.nn.Parameter(torch.Tensor(self.graph_h_dim, 64), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w_q)
        # self.w_k = torch.nn.Parameter(torch.Tensor(self.graph_h_dim, 64), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w_k)
        # self.w_v = torch.nn.Parameter(torch.Tensor(self.graph_h_dim, 64), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w_v)

        # self.loss_r = torch.nn.CrossEntropyLoss()
        # self.loss_e = torch.nn.CrossEntropyLoss()
        self.loss_e = FocalLoss(gamma=2)
        self.loss_r = FocalLoss(gamma=2)

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda)
        #self.classgcn = GraphConv(h_dim, h_dim)
        self.classgcn = TAGConv(h_dim, h_dim)

        self.node2graph = nn.Linear(h_dim, self.graph_h_dim)
        self.node2graph_gate = nn.Sequential(nn.Linear(h_dim, 1), nn.Sigmoid())
        self.reset_gate = nn.Linear(h_dim, 1)
        self.reset_gate1 = nn.Linear(self.num_ents, 1)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        elif decoder_name == "film":
            self.decoder_ob = EventDecoderE(h_dim, num_ents)
            self.rdecoder = EventDecoderR(h_dim, num_rels)
        elif decoder_name =='interacte':
            self.decoder_ob = InteractE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 


    def forward(self, g_list, class_g, use_cuda):
        # gate_list = []              #全是空的
        # degree_list = []            #全是空的

        # gru_hidden = torch.zeros(1, self.h_dim) #GRU隐藏层初值
        # gru_hidden = gru_hidden.to(self.gpu)
        #graph_gru_hidden = torch.zeros(1, self.graph_h_dim)
        #graph_gru_hidden = graph_gru_hidden.to(self.gpu)
        if class_g is None:
            current_ent_emb = self.emb_ent
        else:
            if use_cuda:
                class_g = class_g.to(self.gpu)
            current_ent_emb = self.emb_ent
            current_ent_emb = self.classgcn(class_g, current_ent_emb)
            current_ent_emb = F.normalize(current_ent_emb) if self.layer_norm else current_ent_emb
            self.emb_ent = torch.nn.Parameter(current_ent_emb)
        # current_ent_emb = self.emb_ent
        history_embs = []
        graph_h_list = []
        # graph_h_list = torch.zeros(self.sequence_len, self.graph_h_dim)
        #for i in range(len(g_list)):
        history_graph = merge_graphs(self.num_ents, g_list, use_cuda, self.gpu)
        #history_graph = g_list[0]
        if use_cuda:
            history_graph = history_graph.to(self.gpu)
        new_ent_emb = current_ent_emb
        #for i in range(self.sequence_len):
        new_ent_emb, _ = self.rgcn.forward(history_graph, new_ent_emb, self.emb_rel)
        new_ent_emb = F.normalize(new_ent_emb) if self.layer_norm else new_ent_emb

        # node_emb_graph = self.node2graph(new_ent_emb)
        # node_emb_graph_gate = self.node2graph_gate(new_ent_emb)
        # self.graph_h = torch.sum(torch.mul(node_emb_graph, node_emb_graph_gate), dim=0, keepdim=True)
        # self.graph_h = F.normalize(self.graph_h, dim=1) if self.layer_norm else self.graph_h

        weight_vec = self.reset_gate(new_ent_emb).reshape(1, self.num_ents)
        weight = nn.functional.sigmoid(self.reset_gate1(weight_vec))
        new_ent_emb = new_ent_emb * weight + current_ent_emb * (1 - weight)

        history_embs.append(new_ent_emb)
        graph_h_list.append(self.graph_h)
        # new_rel_emb = torch.mm(self.emb_rel, self.rel_evolve(self.graph_h).reshape(self.h_dim, self.h_dim))

        #print(new_ent_emb)

        return history_embs, self.emb_rel, graph_h_list
        
        # #print(len(new_g_list))
        # for i, g in enumerate(new_g_list):
        #     if use_cuda:
        #         g = g.to(self.gpu)
        #     #print(self.h, self.h_0)
        #     current_ent_emb, current_rel_emb = self.rgcn.forward(g, self.emb_ent, self.emb_rel)  #图上信息传播
        #     current_ent_emb = F.normalize(current_ent_emb) if self.layer_norm else current_ent_emb
        #     self.graph_h = (self.node2graph(current_ent_emb)).sum(0, keepdim=True)      #计算图嵌入向量
        #     self.graph_h = F.normalize(self.graph_h, dim=1)
        #     gru_hidden = self.evolve(self.graph_h, gru_hidden)      #进行状态演化
        #     #print(gru_hidden)
            
        #     evolve_matrix = self.gen_matrix(gru_hidden).reshape(self.h_dim, self.h_dim) #生成状态演化矩阵
        #     evolve_matrix = evolve_matrix + 0.5
        #     current_ent_emb = torch.mm(current_ent_emb, evolve_matrix)      #更新节点状态
        #     current_ent_emb = F.normalize(current_ent_emb) if self.layer_norm else current_ent_emb
        #     if self.encoder_name == 'compgcn':
        #         current_rel_emb = torch.mm(current_rel_emb, evolve_matrix)         #更新边状态
        #     else:
        #         current_rel_emb = torch.mm(self.emb_rel, evolve_matrix)
        #     current_rel_emb = F.normalize(current_rel_emb) if self.layer_norm else current_rel_emb

        #     current_ent_emb = current_ent_emb * self.weight + self.emb_ent * (1 - self.weight)
        #     current_rel_emb = current_rel_emb * self.weight + self.emb_rel * (1 - self.weight)
            
        #     history_embs.append(current_ent_emb)
        #     self.h = current_ent_emb
        #     self.h_0 = current_rel_emb
        # return history_embs, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, test_triplets, class_g, use_cuda):
        with torch.no_grad():
            # inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            # inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            # all_triples = torch.cat((test_triplets, inverse_test_triplets))
            all_triples = test_triplets
            
            evolve_embs, r_emb, graph_embs = self.forward(test_graph, class_g, use_cuda)
            embedding = evolve_embs[-1]
            embedding = F.normalize(embedding) if self.layer_norm else embedding

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, graph_embs[-1],mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, class_g, use_cuda):
        """
        :param glist:
        :param triplets:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_step = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # inverse_triples = triples[:, [2, 1, 0]]
        # inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        # all_triples = torch.cat([triples, inverse_triples])
        all_triples = triples
        all_triples = all_triples.to(self.gpu)

        evolve_embs, r_emb,graph_embs = self.forward(glist, class_g, use_cuda)
        pre_emb = evolve_embs[-1]
        pre_emb = F.normalize(pre_emb) if self.layer_norm else pre_emb

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples, graph_embs[-1]).view(-1, self.num_ents)
            #print(scores_ob)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        return loss_ent, loss_rel
