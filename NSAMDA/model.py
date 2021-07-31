import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import MHATLayer


class HardGAT(nn.Module):
    def __init__(self, G, h_feats, num_heads, num_diseases, num_mrnas, d_sim_dim, m_sim_dim, out_dim,
                 decoder, feat_drop, attn_drop, negative_slope):
        super(HardGAT, self).__init__()

        self.G = G
        self.num_diseases = num_diseases
        self.num_drugs = num_mrnas

        self.gat = MHATLayer(G, h_feats, num_heads, feat_drop, attn_drop, negative_slope)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.m_fc = nn.Linear(h_feats + m_sim_dim, out_dim)    #
        self.d_fc = nn.Linear(h_feats + d_sim_dim, out_dim)    #

        self.BilinearDecoder = BilinearDecoder(feature_size=out_dim)

        self.predict = nn.Sequential(nn.Linear(out_dim * 2, 1), nn.Sigmoid())

        self.decoder = decoder

    def forward(self, G, diseases, mrnas):
        assert G.number_of_nodes() == self.G.number_of_nodes()

        h_agg = self.gat(G)
        h_agg = h_agg.mean(1)
        h_d = torch.cat((h_agg[:self.num_diseases], self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        h_m = torch.cat((h_agg[self.num_diseases:], self.G.ndata['m_sim'][self.num_diseases:]), dim=1)

        h_m = self.feat_drop(F.elu(self.m_fc(h_m)))
        h_d = self.feat_drop(F.elu(self.d_fc(h_d)))

        h = torch.cat((h_d, h_m), dim=0)

        h_diseases = h[diseases]
        h_mrnas = h[mrnas]

        return self.decoder(h_diseases, h_mrnas)


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, h_diseases, h_mrnas):
        x = torch.mul(h_diseases, h_mrnas).sum(1)
        x = torch.reshape(x, [-1])
        outputs = F.sigmoid(x)
        return outputs


class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()
        self.W = Parameter(torch.randn(feature_size, feature_size))

    def forward(self, h_diseases, h_mirnas):
        h_diseases0 = torch.mm(h_diseases, self.W)
        h_mirnas0 = torch.mul(h_diseases0, h_mirnas)
        h0 = h_mirnas0.sum(1)
        h = torch.sigmoid(h0)
        return h
