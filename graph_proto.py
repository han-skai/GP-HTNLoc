import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import pandas as pd
import numpy as np
import data_got
from utils.metric import accuracy
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from dgl.nn.pytorch import MetaPath2Vec


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, bel_out_feats,inc_out_feats):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            'belongs_to': dglnn.GraphConv(in_feats, hid_feats),
            'including': dglnn.GraphConv(in_feats, hid_feats)
        }, aggregate='mean')

        self.conv2 = dglnn.HeteroGraphConv({
            'belongs_to': dglnn.GraphConv(hid_feats, bel_out_feats),
            'including': dglnn.GraphConv(hid_feats, inc_out_feats)
        }, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h



labels = pd.read_csv("data/lncRNA/lncRNA_label.csv", header=None)
features = pd.read_csv("data/bio/bio_feature.csv", header=None)

_, _, base_id = data_got.Bload_data(batch_size=20, num_class=3)
noval_id = data_got.Nload_data(batch_size=20,sample_num=5,samp_freq=30, num_class=3, flag='graph')

f3_labels = labels.iloc[base_id,:3]
f3_features = features.iloc[base_id]

a2_labels = labels.iloc[noval_id,3:]
a2_features = features.iloc[noval_id]



def get_graph(labels, samples_features,label_init_dim):
    rows, cols = np.where(labels.values == 1)
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)

    h_g = dgl.heterograph({
        ('sample', 'belongs_to', 'label'): (rows, cols),
        ('label', 'including', 'sample'): (cols, rows)
    })


    sample_features = torch.tensor(samples_features.values).float()
    node_features = {'sample' : sample_features,
    'label': torch.randn(labels.shape[1], label_init_dim).float()}

    h_g.nodes['sample'].data['feature'] = sample_features
    h_g.nodes['sample'].data['label'] = torch.tensor(labels.values).float()
    h_g.nodes['label'].data['feature'] = torch.randn(labels.shape[1], 51)

    return h_g, node_features




def get_proto(h_g, node_features, in_feats, hid_feats, bel_out_feats,inc_out_feats,iteration_num):
    proto = torch.Tensor()

    for i in range(iteration_num):
        model1 = RGCN(in_feats, hid_feats, bel_out_feats,inc_out_feats)
        criterion1 = nn.BCEWithLogitsLoss()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        for epoch in range(3):
            model1.train()

            logits = model1(h_g, node_features)['sample']

            loss1 = criterion1(logits, h_g.nodes['sample'].data['label'])

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            print(f"第{i}次迭代{loss1.item()}")

        proto = torch.cat([proto, model1(h_g, node_features)['label']], dim=0)

    return proto


def test(h_g,node_features,model):
    with torch.no_grad():
        logits = model(h_g, node_features)['sample']
        predictions = (torch.sigmoid(logits) > 0.5).float()
        labels = h_g.nodes['sample'].data['label']
        test_acc = accuracy(predictions, labels)
        print("test_acc:")
        print(test_acc)


def get_metapath_proto(g, iteration_num):
    model2 = MetaPath2Vec(g, ['belongs_to', 'including','belongs_to', 'including'], window_size=5,emb_dim=156)

    dataloader = DataLoader(torch.arange(g.num_nodes('sample')), batch_size=10,
                            shuffle=False, collate_fn=model2.sample)
    optimizer2 = SparseAdam(model2.parameters(), lr=0.025)

    label_emb = torch.Tensor()
    for i in range(iteration_num):
        for (pos_u, pos_v, neg_v) in dataloader:
            loss2 = model2(pos_u, pos_v, neg_v)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        label_nids = torch.LongTensor(model2.local_to_global_nid['label']).detach()
        label_emb_new = model2.node_embed(label_nids).detach()
        label_emb = torch.cat((label_emb, label_emb_new), dim=0)
    return label_emb

def get_f3_proto():
    f3_g, f3_node_features = get_graph(f3_labels, f3_features, 51)
    f3_proto = get_proto(f3_g, f3_node_features, 51, 128, 256, 3, 60)
    return f3_proto

def get_a2_proto():
    a2_g, a2_node_features = get_graph(a2_labels, a2_features, 51)
    a2_proto = get_proto(a2_g, a2_node_features, 51, 128, 256, 2, 30)
    return a2_proto

def get_f3_meta_graph():
    f3_g, f3_node_features = get_graph(f3_labels, f3_features, 51)
    f3_proto = get_proto(f3_g, f3_node_features, 51, 128, 100, 3, 60)
    f3_mate_proto = get_metapath_proto(f3_g,60)
    f3_proto_all = torch.cat((f3_proto, f3_mate_proto), dim=1)
    return f3_proto_all

def get_a2_meta_graph():
    a2_g, a2_node_features = get_graph(a2_labels, a2_features, 51)
    a2_proto = get_proto(a2_g, a2_node_features, 51, 128, 100, 2, 30)
    a2_mate_proto = get_metapath_proto(a2_g,30)
    a2_proto_all = torch.cat((a2_proto, a2_mate_proto), dim=1)
    return a2_proto_all


if __name__ == '__main__':

    a2p = get_a2_meta_graph()

    print("END")