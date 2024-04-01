import random

from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
from model.batchconv import BatchConv1DLayer
import numpy as np
import sys
path_dir = os.getcwd()

class ConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0):

        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        # if mode=="train":
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        # else:
        #     e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        #     e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_rel.transpose(1, 0))
        return x


'''class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))

        self.in_dim = embedding_dim
        self.kernel_size = kernel_size
        self.channels = channels
        self.conv2 = BatchConv1DLayer(2, channels, padding=int(math.floor(kernel_size / 2)))
        self.scale_w = torch.nn.Parameter(torch.Tensor(embedding_dim*2, (2*self.kernel_size+1)*self.channels))
        self.scale_b = torch.nn.Parameter(torch.Tensor(1,(2*self.kernel_size+1)*self.channels))
        self.shift_w = torch.nn.Parameter(torch.Tensor(embedding_dim*2, (2*self.kernel_size+1)*self.channels))
        self.shift_b = torch.nn.Parameter(torch.Tensor(1,(2*self.kernel_size+1)*self.channels))

        self.theta_e_w = torch.nn.Parameter(torch.Tensor(1, 2*self.channels*self.kernel_size))
        self.theta_e_b = torch.nn.Parameter(torch.Tensor(1, self.channels))
        torch.nn.init.xavier_normal_(self.scale_w)
        torch.nn.init.zeros_(self.scale_b)
        torch.nn.init.xavier_normal_(self.shift_w)
        torch.nn.init.zeros_(self.shift_b)
        torch.nn.init.xavier_normal_(self.theta_e_w)
        torch.nn.init.zeros_(self.theta_e_b)

        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)
    
    def film(self,embedding, w, b):
        #print(embedding.shape, w.shape, b.shape)
        output = torch.mm(embedding, w) + b
        return output[:,:self.channels*self.kernel_size*2], output[:,self.channels*self.kernel_size*2:]

    def forward(self, embedding, emb_rel, triplets, graph_embs, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        #e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        #x = F.relu(x)
        #print(x.shape, e1_embedded_all.shape)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        #print(x.shape)
        #sys.exit()
        return x

    def forward_slow(self, embedding, emb_rel, triplets):
        e1_embedded_all = F.tanh(embedding)
        # e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        # translate to sub space
        # e1_embedded = torch.matmul(e1_embedded, sub_trans)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        e2_embedded = e1_embedded_all[triplets[:, 2]]
        score = torch.sum(torch.mul(x, e2_embedded), dim=1)
        pred = score
        return pred'''

class InteractE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, num_filter=50, kernel_size=3, perm=2, k_h=20, k_w=10, use_bias=True, use_cuda=False, gpu=None):
        super(InteractE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.bn0 = torch.nn.BatchNorm2d(perm)
        #k_h*k_w = embedding_dim
        self.k_h = k_h
        self.k_w = k_w
        self.embedding_dim = embedding_dim
        self.flat_size_h = k_h
        self.flat_size_w = 2*k_w
        self.num_filter = num_filter
        self.perm = perm
        self.padding=0
        self.bn1 = torch.nn.BatchNorm2d(self.num_filter*self.perm)
        self.flat_size = self.flat_size_h * self.flat_size_w * self.num_filter * self.perm
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.fc = torch.nn.Linear(self.flat_size, embedding_dim)
        
        self.kernel_size = kernel_size

        self.register_parameter('bias', Parameter(torch.zeros(num_entities)))
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.num_filter, 1, self.kernel_size, self.kernel_size)))
        torch.nn.init.xavier_normal_(self.conv_filt)
        
        self.gpu=gpu
        self.chequer_perm = self.get_chequer_perm(use_cuda)

    def circular_padding_chw(self, batch, padding):
        upper_pad	= batch[..., -padding:, :]
        lower_pad	= batch[..., :padding, :]
        temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)
        left_pad	= temp[..., -padding:]
        right_pad	= temp[..., :padding]
        padded		= torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None, use_cuda=False):
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]]
        rel_embedded = emb_rel[triplets[:, 1]]
        comb_emb = torch.cat([e1_embedded, rel_embedded], 1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_input = chequer_perm.reshape((-1, self.perm, self.flat_size_w, self.flat_size_h))
        #print(stack_input.shape)
        stack_input = self.bn0(stack_input)
        x = self.inp_drop(stack_input)
        x = self.circular_padding_chw(x, self.kernel_size//2)
        x = F.conv2d(x, self.conv_filt.repeat(self.perm,1,1,1), padding=self.padding, groups=self.perm)
        x = self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_size)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print(batch_size)
        #print(x.shape, e1_embedded_all.shape)
        x = torch.mm(x, e1_embedded_all.transpose(1,0))
        x += self.bias.expand_as(x)
        #print(x.shape)
            
        return x


    def get_chequer_perm(self, use_cuda=False):
            """
            Function to generate the chequer permutation required for InteractE model
            Parameters
            ----------
            
            Returns
            -------
            
            """
            ent_perm  = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
            rel_perm  = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])

            comb_idx = []
            for k in range(self.perm):
                temp = []
                ent_idx, rel_idx = 0, 0

                for i in range(self.k_h):
                    for j in range(self.k_w):
                        if k % 2 == 0:
                            if i % 2 == 0:
                                temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                                temp.append(rel_perm[k, rel_idx]+self.embedding_dim); rel_idx += 1;
                            else:
                                temp.append(rel_perm[k, rel_idx]+self.embedding_dim); rel_idx += 1;
                                temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                        else:
                            if i % 2 == 0:
                                temp.append(rel_perm[k, rel_idx]+self.embedding_dim); rel_idx += 1;
                                temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                            else:
                                temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                                temp.append(rel_perm[k, rel_idx]+self.embedding_dim); rel_idx += 1;

                comb_idx.append(temp)

            chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.gpu) if use_cuda else torch.LongTensor(np.int32(comb_idx))
            return chequer_perm



'''
class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.in_dim = embedding_dim

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        
        self.kernel_size = kernel_size
        self.channels = channels
        self.conv2 = BatchConv1DLayer(2, channels, padding=int(math.floor(kernel_size / 2)))

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

        self.scale_w = torch.nn.Parameter(torch.Tensor(embedding_dim*2, (2*self.kernel_size+1)*self.channels))
        self.scale_b = torch.nn.Parameter(torch.Tensor(1,(2*self.kernel_size+1)*self.channels))
        self.shift_w = torch.nn.Parameter(torch.Tensor(embedding_dim*2, (2*self.kernel_size+1)*self.channels))
        self.shift_b = torch.nn.Parameter(torch.Tensor(1,(2*self.kernel_size+1)*self.channels))

        self.theta_e_w = torch.nn.Parameter(torch.Tensor(1, 2*self.channels*self.kernel_size))
        self.theta_e_b = torch.nn.Parameter(torch.Tensor(1, self.channels))
        self.history_len=5
        self.bn_out = torch.nn.BatchNorm1d(self.history_len)

        torch.nn.init.xavier_normal_(self.scale_w)
        torch.nn.init.zeros_(self.scale_b)
        torch.nn.init.xavier_normal_(self.shift_w)
        torch.nn.init.zeros_(self.shift_b)
        torch.nn.init.xavier_normal_(self.theta_e_w)
        torch.nn.init.zeros_(self.theta_e_b)
    
    def film(self,embedding, w, b):
        #print(embedding.shape, w.shape, b.shape)
        output = torch.mm(embedding, w) + b
        return output[:,:self.channels*self.kernel_size*2], output[:,self.channels*self.kernel_size*2:]

    def forward(self, embedding, emb_rel, triplets, graph_emb, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        #for embedding, graph_emb in zip(embedding_list, graph_emb_list):
        embedding = F.normalize(embedding)
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        # print(self.scale_w.shape, self.scale_b.shape, self.shift_w.shape, self.shift_b.shape)
        alpha_w, alpha_b = self.film(graph_emb, self.scale_w, self.scale_b)
        #(out_channel * (in_channel*kernel_size+1)) = (in*2)*(in*2, out_channel * (in_channel*kernel_size+1)) + (1, out_channel * (in_channel*kernel_size+1))
        beta_w, beta_b = self.film(graph_emb, self.shift_w, self.shift_b)
        #(out_channel * (in_channel*kernel_size+1)) = (in*2)*(in*2, out_channel * (in_channel*kernel_size+1)) + (1, out_channel * (in_channel*kernel_size+1))
        # print((alpha_w + 1).shape, self.theta_e_w.repeat(batch_size, 1).shape, (alpha_b + 1).shape, self.theta_e_b.repeat(batch_size, 1).shape)

        new_theta_e_w = torch.mul((alpha_w + 1), self.theta_e_w) + beta_w
        #(in_channel * out_channel * kernel_size) = (in_channel * out_channel * kernel_size)*(in_channel * out_channel * kernel_size) + (in_channel * out_channel * kernel_size)
        new_theta_e_b = torch.mul((alpha_b + 1), self.theta_e_b) + beta_b
        #(out_channel) = (out_channel) * (out_channel) + (out_channel)
        # print(new_theta_e_w.shape, new_theta_e_b.shape)
        #print(x.shape)

        x = F.conv1d(x, new_theta_e_w.view(self.channels, 2, self.kernel_size), new_theta_e_b.view(self.channels), padding=int(math.floor(self.kernel_size / 2)))
        
        #x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        
        #x = torch.bmm((e1_embedded_all[triplets[:, 0]] + emb_rel[triplets[:, 1]]).pow(2).unsqueeze(1), new_theta_e_w.reshape(batch_size, self.in_dim, self.in_dim)).squeeze(1) + new_theta_e_b
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
            # if ans is None:
            #     ans = x.unsqueeze(1)
            # else:
            #     ans = torch.cat([ans,x.unsqueeze(1)], dim=1)
        # if ans.shape[1] != self.history_len:
        #     ans = ans[:, -1, :].squeeze(1)
        # else:
        #     ans = self.bn_out(ans)
        # ans = torch.mean(ans, dim=1)
        return x'''

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=2, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.in_dim = embedding_dim

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        
        self.kernel_size = kernel_size
        self.channels = channels
        self.conv2 = BatchConv1DLayer(2, channels, padding=int(math.floor(kernel_size / 2)))

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

        self.scale_w = torch.nn.Parameter(torch.Tensor(embedding_dim*2, (2*self.kernel_size+1)*self.channels))
        self.scale_b = torch.nn.Parameter(torch.Tensor(1,(2*self.kernel_size+1)*self.channels))
        self.shift_w = torch.nn.Parameter(torch.Tensor(embedding_dim*2, (2*self.kernel_size+1)*self.channels))
        self.shift_b = torch.nn.Parameter(torch.Tensor(1,(2*self.kernel_size+1)*self.channels))

        self.theta_e_w = torch.nn.Parameter(torch.Tensor(1, 2*self.channels*self.kernel_size))
        self.theta_e_b = torch.nn.Parameter(torch.Tensor(1, self.channels))
        self.history_len=5
        self.bn_out = torch.nn.BatchNorm1d(self.history_len)

        torch.nn.init.xavier_normal_(self.scale_w)
        torch.nn.init.zeros_(self.scale_b)
        torch.nn.init.xavier_normal_(self.shift_w)
        torch.nn.init.zeros_(self.shift_b)
        torch.nn.init.xavier_normal_(self.theta_e_w)
        torch.nn.init.zeros_(self.theta_e_b)
    
    def film(self,embedding, w, b):
        #print(embedding.shape, w.shape, b.shape)
        output = torch.mm(embedding, w) + b
        return output[:,:self.channels*self.kernel_size*2], output[:,self.channels*self.kernel_size*2:]

    def forward(self, embedding, emb_rel, triplets, graph_emb, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        #for embedding, graph_emb in zip(embedding_list, graph_emb_list):
        embedding = F.normalize(embedding)
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        # print(self.scale_w.shape, self.scale_b.shape, self.shift_w.shape, self.shift_b.shape)
        alpha_w, alpha_b = self.film(torch.cat([e1_embedded_all[triplets[:, 0]], emb_rel[triplets[:, 1]]], dim=1), self.scale_w, self.scale_b)
        #(batch_size, out_channel * (in_channel*kernel_size+1)) = (batch_size, in*2)*(in*2, out_channel * (in_channel*kernel_size+1)) + (1, out_channel * (in_channel*kernel_size+1))
        beta_w, beta_b = self.film(torch.cat([e1_embedded_all[triplets[:, 0]], emb_rel[triplets[:, 1]]], dim=1), self.shift_w, self.shift_b)
        #(batch_size, out_channel * (in_channel*kernel_size+1)) = (batch_size, in*2)*(in*2, out_channel * (in_channel*kernel_size+1)) + (1, out_channel * (in_channel*kernel_size+1))
        # print((alpha_w + 1).shape, self.theta_e_w.repeat(batch_size, 1).shape, (alpha_b + 1).shape, self.theta_e_b.repeat(batch_size, 1).shape)

        new_theta_e_w = torch.mul((alpha_w + 1), self.theta_e_w.repeat(batch_size, 1)) + beta_w
        #(batch_size, in_channel * out_channel * kernel_size) = (batch_size, in_channel * out_channel * kernel_size)*(batch_size, in_channel * out_channel * kernel_size) + (batch_size, in_channel * out_channel * kernel_size)
        new_theta_e_b = torch.mul((alpha_b + 1), self.theta_e_b.repeat(batch_size, 1)) + beta_b
        #(batch_size, out_channel) = (batch_size, out_channel) * (batch_size, out_channel) + (batch_size, out_channel)
        # print(new_theta_e_w.shape, new_theta_e_b.shape)
        #print(x.shape)
        x = self.conv2(x.unsqueeze(1), new_theta_e_w.view(batch_size, self.channels, 2, self.kernel_size), new_theta_e_b)
        x = x.squeeze(1).view(-1,self.channels*self.in_dim)
        #print(x.shape)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
            # if ans is None:
            #     ans = x.unsqueeze(1)
            # else:
            #     ans = torch.cat([ans,x.unsqueeze(1)], dim=1)
        # if ans.shape[1] != self.history_len:
        #     ans = ans[:, -1, :].squeeze(1)
        # else:
        #     ans = self.bn_out(ans)
        # ans = torch.mean(ans, dim=1)
        return x
    
    def forward_raw(self, embedding, emb_rel, triplets, graph_embs, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        #e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        #x = F.relu(x)
        #print(x.shape, e1_embedded_all.shape)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        #print(x.shape)
        #sys.exit()
        return x