import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.sampling import select_topk
from functools import partial
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
from dgl.base import DGLError
import dgl

class NSGAT(nn.Module):
    def __init__(self, G,
                 out_feats,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 k=8,
                 activation=F.elu):
        super(NSGAT, self).__init__()
        self.G = G
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mrna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.k = k

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], out_feats * num_heads, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self.out_feats)))

        self.pr = nn.Parameter(torch.FloatTensor(size=(1,G.ndata['m_sim'].shape[1])))
        self.pi = nn.Parameter(torch.FloatTensor(size=(1, G.ndata['d_sim'].shape[1])))
        # Initialize Dropouts
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.pr, gain=gain)
        nn.init.xavier_normal_(self.pi, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, get_attention=False):
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # Check in degree and generate error
            if (graph.in_degrees()==0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            # projection process to get importance vector y
            graph.ndata['yl'] = torch.abs(torch.matmul(self.pr, graph.ndata['m_sim'].T)
                                              .view(-1))/torch.norm(self.pr,p=2)
            graph.ndata['yr'] = torch.abs(torch.matmul(self.pi, graph.ndata['d_sim'].T)
                                              .view(-1)) / torch.norm(self.pi, p=2)

            # Use edge message passing function to get the weight from src node
            graph.apply_edges(fn.copy_u('yl','yl'))
            #graph.apply_edges(fn.copy_u('yr', 'yr'))
            # Select Top k neighbors
            subgraph = select_topk(graph.cpu(),self.k,'yl').to(graph.device)
            # Sigmoid as information threshold
            subgraph.ndata['yl'] = torch.sigmoid(subgraph.ndata['yl'])
            subgraph.ndata['yr'] = torch.sigmoid(subgraph.ndata['yr'])
            # Using vector matrix elementwise mul for acceleration
            m_feat = subgraph.ndata['yl'].view(-1,1)*graph.ndata['m_sim']
            d_feat = subgraph.ndata['yr'].view(-1, 1) * graph.ndata['d_sim']
            m_feat = self.feat_drop(m_feat)
            d_feat = self.feat_drop(d_feat)
            h_m = self.m_fc(m_feat).view(-1, self.num_heads, self.out_feats)
            h_d = self.d_fc(d_feat).view(-1, self.num_heads, self.out_feats)
            el = (h_m * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (h_d * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # Assign the value on the subgraph
            subgraph.srcdata.update({'ft': h_m, 'el': el})
            subgraph.dstdata.update({'er': er})

            assert graph.number_of_nodes() == self.G.number_of_nodes()
            self.G.apply_nodes(lambda nodes: {'z': self.attn_drop(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            self.G.apply_nodes(lambda nodes: {'z': self.attn_drop(self.m_fc(nodes.data['m_sim']))}, self.mrna_nodes)

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            subgraph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(subgraph.edata.pop('e'))
            # compute softmax
            subgraph.edata['a'] = self.attn_drop(edge_softmax(subgraph, e))
            # message passing
            subgraph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = subgraph.dstdata['ft']
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, subgraph.edata['a']
            else:
                return rst


class MHATLayer(nn.Module):
    def __init__(self, G, out_feats, num_heads, feat_drop, attn_drop, slope, merge='cat'):
        super(MHATLayer, self).__init__()

        self.G = G
        self.slope = slope
        self.merge = merge

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(NSGAT(G, out_feats, num_heads, feat_drop, attn_drop, slope))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)
