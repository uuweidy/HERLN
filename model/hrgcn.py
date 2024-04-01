from cmath import exp
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class HawkesRGCNLayer(nn.Module):
	def __init__(self, in_feat, out_feat, num_rels, 
					activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
		super(HawkesRGCNLayer, self).__init__()

		self.in_feat = in_feat
		self.out_feat = out_feat
		self.activation = activation
		self.self_loop = self_loop
		self.num_rels = num_rels
		self.rel_emb = None
		self.skip_connect = skip_connect

		# WL
		self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat*2, self.out_feat))
		nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

		self.attn_fc = nn.Linear(3 * out_feat, 1, bias=False)
		# k,用于计算时间衰减
		self.delta = nn.Parameter(torch.ones(*[1]), requires_grad=True).float()

		if self.self_loop:
			# 分别处理独立节点和有连接的节点（是否需要？）
			self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
			nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
			self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
			nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

		if self.skip_connect:
			self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
			nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
			self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
			nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

		if dropout:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None

	def propagate(self, g):
		#g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

		#这样是不是不再需要apply function?
		g.update_all(lambda x: self.msg_func(x), self.reduce_func, self.apply_func)

	def forward(self, g, prev_h, emb_rel):
		self.rel_emb = emb_rel
		# self.sub = sub
		# self.ob = ob
		if self.self_loop:
			masked_index = torch.masked_select(
				torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
				(g.in_degrees(range(g.number_of_nodes())) > 0))
			loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
			loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
		if len(prev_h) != 0 and self.skip_connect:
			skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

		# calculate the neighbor message with weight_neighbor
		g.apply_edges(self.edge_attention)
		self.propagate(g)
		node_repr = g.ndata['h']
		# print(len(prev_h))
		if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
			if self.self_loop:
				node_repr = node_repr + loop_message
			node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
		else:
			if self.self_loop:
				node_repr = node_repr + loop_message

		if self.activation:
			node_repr = self.activation(node_repr)
		if self.dropout is not None:
			node_repr = self.dropout(node_repr)
		g.ndata['h'] = node_repr
		return node_repr


	def edge_attention(self, edges):
		z2 = torch.cat([edges.src['h'], edges.dst['h'], self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)], dim=1)
		a = self.attn_fc(z2)
		return {'e': F.leaky_relu(a)}


	def msg_func(self, edges):
		# if reverse:
		#     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
		# else:
		#     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
		relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
		edge_type = edges.data['type']
		edge_time = edges.data['time']
		edge_num = edge_type.shape[0]

		k = (-1*edge_time*self.delta).reshape(-1,1)
		node = edges.src['h'].view(-1, self.out_feat)
		# node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
		#                  torch.matmul(node[edge_num // 2:, :], self.ob)])
		# node = torch.matmul(node, self.sub)

		# after add inverse edges, we only use message pass when h as tail entity
		# 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
		msg = torch.cat((node, relation), dim=1)
		#msg = node + relation
		# calculate the neighbor message with weight_neighbor
		msg = torch.mm(msg, self.weight_neighbor)
		#print(msg)
		#print(k)
		return {'msg': msg, 'k': k, 'e': edges.data['e']}


	def reduce_func(self, nodes):
		k = self.dropout(F.softmax(torch.mul(nodes.mailbox['k'], nodes.mailbox['e']), dim=1))
		h = torch.sum(k*nodes.mailbox['msg'], dim=1)
		return {'h': h}

	def apply_func(self, nodes):
		return {'h': nodes.data['h']}