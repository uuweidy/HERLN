import random

from torch.nn import functional as F
import torch
import math
import os
path_dir = os.getcwd()

class EventDecoderR(torch.nn.Module):
	def __init__(self, in_dim, out_dim):			# in是节点维度，out是最后的分类个数
		super(EventDecoderR, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim*2
		self.hid_dim = (in_dim+1)*1

		self.scale_w = torch.nn.Parameter(torch.Tensor(in_dim*2, self.hid_dim))
		self.scale_b = torch.nn.Parameter(torch.Tensor(1,self.hid_dim))
		self.shift_w = torch.nn.Parameter(torch.Tensor(in_dim*2, self.hid_dim))
		self.shift_b = torch.nn.Parameter(torch.Tensor(1,self.hid_dim))

		self.theta_e_w = torch.nn.Parameter(torch.Tensor(1, in_dim))
		self.theta_e_b = torch.nn.Parameter(torch.Tensor(1,1))

		self.classfier = torch.nn.Linear(in_dim, out_dim*2)

		torch.nn.init.xavier_normal_(self.scale_w)
		torch.nn.init.zeros_(self.scale_b)
		torch.nn.init.xavier_normal_(self.shift_w)
		torch.nn.init.zeros_(self.shift_b)
		torch.nn.init.xavier_normal_(self.theta_e_w)
		torch.nn.init.zeros_(self.theta_e_b)

	def film(self,embedding, w, b):
		#print(embedding.shape, w.shape, b.shape)
		output = torch.mm(embedding, w) + b
		return output[:,:self.in_dim], output[:,self.in_dim:]
	
	def forward(self, ent_emb, rel_emb, triples, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
		batch_size = len(triples)
		# if mode=="train":
		#print(triples)
		s_embedded = ent_emb[triples[:, 0]]
		#r_embedded = rel_emb[triples[:, 1]]
		o_embedded = ent_emb[triples[:, 2]]
		alpha_w, alpha_b = self.film(torch.cat([s_embedded, o_embedded], dim=1), self.scale_w, self.scale_b)
		beta_w, beta_b = self.film(torch.cat([s_embedded, o_embedded], dim=1), self.shift_w, self.shift_b)

		#print((alpha_w + 1).shape, self.theta_e_w.repeat(batch_size, 1).shape)

		new_theta_e_w = torch.mul((alpha_w + 1), self.theta_e_w.repeat(batch_size, 1)) + beta_w
		new_theta_e_b = torch.mul((alpha_b + 1), self.theta_e_b.repeat(batch_size, 1)) + beta_b
		#print(new_theta_e_w.shape, new_theta_e_b.shape)
		lambda_t = (s_embedded - o_embedded).pow(2) * new_theta_e_w + new_theta_e_b

		#print(lambda_t.shape)
		lambda_t = self.classfier(lambda_t)

		return lambda_t

class EventDecoderE(torch.nn.Module):
	def __init__(self, in_dim, out_dim):			# in是节点维度，out是最后的分类个数
		super(EventDecoderE, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hid_dim = (in_dim+1)*in_dim

		# self.scale_w = torch.nn.Parameter(torch.Tensor(in_dim*2, self.hid_dim))
		# self.scale_b = torch.nn.Parameter(torch.Tensor(1,self.hid_dim))
		# self.shift_w = torch.nn.Parameter(torch.Tensor(in_dim*2, self.hid_dim))
		# self.shift_b = torch.nn.Parameter(torch.Tensor(1,self.hid_dim))

		# self.theta_e_w = torch.nn.Parameter(torch.Tensor(1, in_dim))
		# self.theta_e_b = torch.nn.Parameter(torch.Tensor(1,1))

		self.scale_w = torch.nn.Parameter(torch.Tensor(in_dim*2, self.hid_dim))
		self.scale_b = torch.nn.Parameter(torch.Tensor(1,self.hid_dim))
		self.shift_w = torch.nn.Parameter(torch.Tensor(in_dim*2, self.hid_dim))
		self.shift_b = torch.nn.Parameter(torch.Tensor(1,self.hid_dim))

		self.theta_e_w = torch.nn.Parameter(torch.Tensor(1, in_dim*in_dim))
		self.theta_e_b = torch.nn.Parameter(torch.Tensor(1, in_dim))

		self.classfier = torch.nn.Linear(in_dim, out_dim)

		torch.nn.init.xavier_normal_(self.scale_w)
		torch.nn.init.zeros_(self.scale_b)
		torch.nn.init.xavier_normal_(self.shift_w)
		torch.nn.init.zeros_(self.shift_b)
		torch.nn.init.xavier_normal_(self.theta_e_w)
		torch.nn.init.zeros_(self.theta_e_b)

	def film(self,embedding, w, b):
		#print(embedding.shape, w.shape, b.shape)
		output = torch.mm(embedding, w) + b
		return output[:,:self.in_dim*self.in_dim], output[:,self.in_dim*self.in_dim:]
	
	def forward(self, ent_emb, rel_emb, triples, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
		batch_size = len(triples)
		# if mode=="train":
		#print(triples)
		s_embedded = ent_emb[triples[:, 0]]
		r_embedded = rel_emb[triples[:, 1]]
		#o_embedded = ent_emb[triples[:, 2]]
		# print(self.scale_w.shape, self.scale_b.shape, self.shift_w.shape, self.shift_b.shape)
		alpha_w, alpha_b = self.film(torch.cat([s_embedded, r_embedded], dim=1), self.scale_w, self.scale_b)
		#(batch_size, in * (out+1)) = (batch_size, in*2)*(in*2, in*(out+1)) + (1, in*(out+1))
		beta_w, beta_b = self.film(torch.cat([s_embedded, r_embedded], dim=1), self.shift_w, self.shift_b)
		#(batch_size, in * (out+1)) = (batch_size, in*2)*(in*2, in*(out+1)) + (1, in*(out+1))
		# print((alpha_w + 1).shape, self.theta_e_w.repeat(batch_size, 1).shape, (alpha_b + 1).shape, self.theta_e_b.repeat(batch_size, 1).shape)

		new_theta_e_w = torch.mul((alpha_w + 1), self.theta_e_w.repeat(batch_size, 1)) + beta_w
		#(batch_size, in * out) = (batch_size, in * out)*(batch_size, in * out) + (batch_size, in * out)
		new_theta_e_b = torch.mul((alpha_b + 1), self.theta_e_b.repeat(batch_size, 1)) + beta_b
		#(batch_size, in) = (batch_size, in) * (batch_size, in) + (batch_size, in)
		# print(new_theta_e_w.shape, new_theta_e_b.shape)
		
		lambda_t = torch.bmm((s_embedded + r_embedded).pow(2).unsqueeze(1),new_theta_e_w.reshape(batch_size, self.in_dim, self.in_dim)).squeeze(1) + new_theta_e_b

		#lambda_t = (s_embedded + r_embedded).pow(2) * new_theta_e_w + new_theta_e_b

		# print(lambda_t.shape)
		lambda_t = self.classfier(lambda_t)

		return lambda_t