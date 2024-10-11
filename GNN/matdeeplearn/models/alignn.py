import torch
from torch import Tensor 
import torch.nn.functional as F 
from torch.nn import Sequential, Linear, BatchNorm1d
import torch_geometric 
from torch_geometric.nn import (
	Set2Set,
	global_mean_pool,
	global_add_pool,
	global_max_pool,
	ResGatedGraphConv
	)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter 

#ALIGNN 
class ALIGNN(torch.nn.Module):
	def __init__(
		self,
		data,
		dim1=64, 
		dim2=64,
		pre_fc_count=1,
		gc_count=3,
		post_fc_count=1,
		pool='global_mean_pool',
		pool_order='early',
		batch_norm='True',
		batch_track_stats='True',
		act='silu',
		dropout_rate=0.0,
		**kwargs
		):
		super(ALIGNN, self).__init__()

		if batch_track_stats == 'False':
			self.batch_track_stats = False
		else:
			self.batch_track_stats = True 
		self.batch_norm = batch_norm 
		self.pool = pool 
		self.act = act 
		self.pool_order = pool_order
		self.dropout_rate = dropout_rate

		#determine gc dimension
		assert gc_count > 0, 'Need at least 1 GC layer'
		if pre_fc_count == 0:
			gc_dim = data.num_features
		else:
			gc_dim = dim1
		#I think this is a bug in the original code, fix that
		if post_fc_count == 0:
			post_fc_dim = data.num_features
		else:
			post_fc_dim = dim1 
		#determine output dimension length
		output_dim = 1

		#set up pre-GNN dense layers
		if pre_fc_count > 0:
			self.pre_lin_list = torch.nn.ModuleList()
			for i in range(pre_fc_count):
				if i == 0:
					lin = torch.nn.Linear(data.num_features, dim1)
					self.pre_lin_list.append(lin)
				else:
					lin = torch.nn.Linear(dim1, dim1)
					self.pre_lin_list.append(lin)
		elif pre_fc_count == 0:
			self.pre_lin_list = torch.nn.ModuleList()

		#set up GNN layers
		self.conv_list = torch.nn.ModuleList()
		self.bn_list = torch.nn.ModuleList()
		gc_activation = getattr(F, self.act)
		for i in range(gc_count):
			#let's just start here and see if this works
			#the parameters are:
				#inchannels
				#outchannels
				#activation function
				#edge_dim (data.edge_attr[1])
			conv = ResGatedGraphConv(
				in_channels = gc_dim,
				out_channels = gc_dim,
				act = gc_activation,
				edge_dim = data.num_edge_features
				)
			self.conv_list.append(conv)
			if self.batch_norm == 'True':
				bn = BatchNorm1d(gc_dim, track_running_stats = self.batch_track_stats)
				self.bn_list.append(bn)

		#set up post-GNN layers 
		if post_fc_count > 0:
			self.post_lin_list = torch.nn.ModuleList()
			for i in range(post_fc_count):
				if i == 0:
					if self.pool_order == 'early' and self.pool == 'set2set':
						lin = torch.nn.Linear(post_fc_dim * 2, dim2)
					else:
						lin = torch.nn.Linear(post_fc_dim, dim2)
					self.post_lin_list.append(lin)
				else:
					lin = torch.nn.Linear(dim2, dim2)
					self.post_lin_list.append(lin)
			self.lin_out = torch.nn.Linear(dim2, output_dim)

		elif post_fc_count == 0:
			self.post_lin_list = torch.nn.ModuleList()
			if self.pool_order == 'early' and self.pool == 'set2set':
				self.lin_out = torch.nn.Linear(post_fc_dim * 2, output_dim)
			else:
				self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)

		if self.pool_order == 'early' and self.pool == 'set2set':
			self.set2set = Set2Set(post_fc_dim, processing_steps=3)
		else:
			self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
			self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

	def forward(self, data):

		#pre-GNN dense layers
		for i in range(0, len(self.pre_lin_list)):
			if i == 0:
				out = self.pre_lin_list[i](data.x)
				out = getattr(F, self.act)(out)
			else:
				out = self.pre_lin_list[i](data.x)
				out = getattr(F, self.act)(out)

		#GNN layers
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

		#Post GNN dense layers
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
