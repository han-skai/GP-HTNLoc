import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import pandas as pd
import numpy as np
import data_got
from utils.metric import accuracy


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, bel_out_feats,inc_out_feats):
        super().__init__()
        # Create different convolutional layers for different relationship types
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


labels = pd.read_csv("data/Benchmark Dataset/lncRNA_label.csv", header=None)
features = pd.read_csv("data/Benchmark Dataset/feature/lncRNA_feature.csv", header=None)


base_transf, base_sam_all,base_sam_unique= data_got.one_sample_base2avg(batch_size=20, sample_num=5,samp_freq=60, num_class=3)
novel_sam_all, novel_sam_unique = data_got.Nload_data(batch_size=20,sample_num=5,samp_freq=30, num_class=3, flag='graph_bilstm')




f3_labels = labels.iloc[base_sam_all, :3]
f3_features = features.iloc[base_sam_all]

a2_labels = labels.iloc[novel_sam_all, 3:]
a2_features = features.iloc[novel_sam_all]


def get_graph(labels, samples_features,label_init_dim):
    rows, cols = np.where(labels.values == 1)
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)

    h_g = dgl.heterograph({
        ('sample', 'belongs_to', 'label'): (rows, cols),
        ('label', 'including', 'sample'): (cols, rows)
    })


    sample_features = samples_features.float()
    node_features = {'sample' : sample_features,
    'label': torch.randn(labels.shape[1], label_init_dim).float()}

    # Establish a node feature dictionary on the graph
    h_g.nodes['sample'].data['feature'] = sample_features
    h_g.nodes['sample'].data['label'] = torch.tensor(labels.values).float()
    h_g.nodes['label'].data['feature'] = torch.randn(labels.shape[1], 256)

    return h_g, node_features




def get_proto(h_g, node_features, in_feats, hid_feats, bel_out_feats,inc_out_feats,iteration_num):
    proto = torch.Tensor()

    for i in range(iteration_num):
        model = RGCN(in_feats, hid_feats, bel_out_feats,inc_out_feats)
        criterion1 = nn.BCEWithLogitsLoss()
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(3):
            model.train()

            logits = model(h_g, node_features)['sample']
            loss1 = criterion1(logits, h_g.nodes['sample'].data['label'])

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            print(f"{i}-th iteration{loss1.item()}")
        proto = torch.cat([proto, model(h_g, node_features)['label']], dim=0)
    return proto


def test(h_g,node_features,model):
    with torch.no_grad():
        logits = model(h_g, node_features)['sample']
        predictions = (torch.sigmoid(logits) > 0.5).float()
        labels = h_g.nodes['sample'].data['label']
        test_acc = accuracy(predictions, labels)
        print("test_acc:")
        print(test_acc)




if __name__ == '__main__':
    print("END")
