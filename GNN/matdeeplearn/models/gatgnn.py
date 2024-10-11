#GATGNN imports
import numpy as np 
import torch
from torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn 

from torch_geometric.nn.conv import MessagePassing 
from torch_geometric.utils import softmax 
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add 
from torch_geometric.nn.inits import glorot, zeros

#MatDeepLearn imports
from torch import Tensor 
from torch.nn import Sequential, BatchNorm1d
import torch_geometric 
from torch_geometric.nn import (
	Set2Set,
	global_mean_pool,
	global_add_pool,
	global_max_pool
	)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter 

class GATGNN(torch.nn.Module):
	def __init__(
		self,
		data,
		dim1=64,
		dim2=64, 
		dim3=1, #the number of heads that are in the attention mechanism
		pre_fc_count=1,
		gc_count=3,
		post_fc_count=1,
		pool='global_mean_pool',
		pool_order='early',
		batch_norm='True',
		batch_track_stats='True',
		act='relu',
		dropout_rate=0.0,
		**kwargs
	):
		super(GATGNN, self).__init__()

		if batch_track_stats == 'False':
			self.batch_track_stats = False 
		else:
			self.batch_track_stats = True 
		self.batch_norm = batch_norm
		self.pool = pool 
		self.act = act 
		self.pool_order = pool_order
		self.dropout_rate = dropout_rate
		self.heads = dim3

	#in the code, n_h = number of neurons
	#GAT_Crystal(n_h, n_h, n_h, self.n_heads) for i in range(layers)
	#in_features = out_features = edge_dim 
	#I think we can pass our own edge_dim, might have to look at previous code for this implementation


		assert gc_count > 0, 'Need at least 1 GC layer'
		if pre_fc_count == 0:
			gc_dim = data.num_features
		else:
			gc_dim = dim1

		if post_fc_count == 0:
			post_fc_dim = data.num_features 
		else:
			post_fc_dim = dim1 

		output_dim = 1

		if pre_fc_count > 0:
			self.pre_lin_list = torch.nn.ModuleList()
			for i in range(pre_fc_count):
				if i == 0:
					lin = Linear(data.num_features, dim1)
					self.pre_lin_list.append(lin)
				else:
					lin = Linear(dim1, dim1)
					self.pre_lin_list.append(lin)
		elif pre_fc_count == 0:
			self.pre_lin_list = torch.nn.ModuleList()

		#GNN layers 
		self.conv_list = torch.nn.ModuleList()
		self.bn_list = torch.nn.ModuleList()

		for i in range(gc_count):
			conv = GATConv(in_features=gc_dim, out_features=gc_dim, edge_dim = data.num_edge_features, heads=self.heads)
			self.conv_list.append(conv)
			if self.batch_norm == 'True':
				bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
				self.bn_list.append(bn)

		if post_fc_count > 0:
			self.post_lin_list = torch.nn.ModuleList()
			for i in range(post_fc_count):
				if i == 0:
					if self.pool_order == 'early' and self.pool == 'set2set':
						lin = Linear(post_fc_dim * 2, dim2)
					else:
						lin = Linear(post_fc_dim, dim2)
					self.post_lin_list.append(lin)
				else:
					lin = Linear(dim2, dim2)
					self.post_lin_list.append(lin)
			self.lin_out = Linear(dim2, output_dim)

		elif post_fc_count == 0:
			self.post_lin_list = torch.nn.ModuleList()
			if self.pool_order == 'early' and self.pool == 'set2set':
				self.lin_out = Linear(post_fc_dim * 2, output_dim)
			else:
				self.lin_out = Linear(post_fc_dim, output_dim)

		if self.pool_order == 'early' and self.pool == 'set2set':
			self.set2set = Set2Set(post_fc_dim, processing_steps=3)
		elif self.pool_order == 'late' and self.pool == 'set2set':
			self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
			self.lin_out_2 = Linear(output_dim * 2, output_dim)

	def forward(self, data):

		for i in range(0, len(self.pre_lin_list)):
			if i == 0:
				out = self.pre_lin_list[i](data.x)
				out = getattr(F, self.act)(out)
			else:
				out = self.pre_lin_list[i](out)
				out = getattr(F, self.act)(out)

		#the original code has embedding layers (linear layers) before this step
		#however we take of that with the pre_lin_list functions
		#this somewhat resembles the GATConv layer from pytorch_geometric
		#well maybe not
		#data.x = Linear(92, 100)(out)
		#data.edge_attr = Linear(41, 100)
		data.edge_attr  = F.leaky_relu(data.edge_attr, 0.2) #neg_slope is just 0.2

		for i in range(0, len(self.conv_list)):
			if len(self.pre_lin_list) == 0 and i == 0:
				if self.batch_norm == 'True':
					out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
					out = self.bn_list[i](out)
				else:
					out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
			else:
				if self.batch_norm == 'True':
					out = self.conv_list[i](out, data.edge_index, data.edge_attr)
					out = self.bn_list[i](out)
				else:
					out = self.conv_list[i](out, data.edge_index, data.edge_attr)
			out = F.dropout(out, p=self.dropout_rate, training=self.training)


		if self.pool_order == 'early':
			if self.pool == 'set2set':
				out = self.set2set(out, data.batch)
			else:
				out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
			for i in range(0, len(self.post_lin_list)):
				out = self.post_lin_list[i](out)
				out = getattr(F, self.act)(out)
			out = self.lin_out(out)

		elif self.pool_order == 'late':
			for i in range(0, len(self.post_lin_list)):
				out = self.post_lin_list[i](out)
				out = getattr(F, self.act)(out)
			out = self.lin_out(out)
			if self.pool == 'set2set':
				out = self.set2set(out, data.batch)
				out = self.lin_out_2(out)
			else:
				out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

		if out.shape[1] == 1:
			return out.view(-1)
		else:
			return out


class GATConv(MessagePassing):
	def __init__(
		self,
		in_features,
		out_features,
		edge_dim, 
		heads,
		**kwargs  
	):
		#need to add some aggr=mean argument
		super(GATConv, self).__init__(aggr='mean', **kwargs)
		self.in_features = in_features
		self.out_features = out_features
		self.heads = heads 
		#might need a parameter for the activation function here to be passed from the other class
		#they handcraft these layers it seems
		#the weights
		self.W = Parameter(Tensor(in_features+edge_dim, heads*out_features))
		self.att = Parameter(Tensor(1, heads, 2*out_features))
		#the bias
		self.bias = Parameter(Tensor(out_features)) 
		self.reset_parameters()

	def reset_parameters(self):
		glorot(self.W)
		glorot(self.att)
		zeros(self.bias)

	def forward(self, x, edge_index, edge_attr):
		#print('Output of the convolution layer', self.propagate(edge_index, x=x, edge_attr=edge_attr))
		return self.propagate(edge_index, x=x, edge_attr=edge_attr)

	def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
		x_i = torch.cat([x_i, edge_attr], dim=-1)
		x_j = torch.cat([x_j, edge_attr], dim=-1)

		x_i = F.softplus(torch.matmul(x_i, self.W))
		x_j = F.softplus(torch.matmul(x_j, self.W))
		x_i = x_i.view(-1, self.heads, self.out_features)
		x_j = x_j.view(-1, self.heads, self.out_features)

		#this self.att thing needs to be implemented in order for this to work
		alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))
		#batch normalization normally done here in the original code
		alpha = softmax(alpha, edge_index_i)
		#dropout is done here but we're going to do it in the other class to match the other code templates
		x_j = (x_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
		return x_j

	def update(self, aggr_out, x):
		aggr_out = aggr_out.mean(dim=0)
		aggr_out += self.bias 
		return aggr_out
