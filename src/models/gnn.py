import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GraphRanker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['gnn']['hidden_dim']
        self.dropout = config['gnn']['dropout']
        self.heads = config['gnn']['heads']
        self.input_dim = config['embeddings']['embedding_dim']
        
        self.edge_emb = nn.Embedding(2, 64)

        #query conditioned vector gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
        )

        self.conv1 = GATv2Conv(self.input_dim * 3, self.hidden_dim, heads=self.heads, dropout=self.dropout, edge_dim=64)
        self.conv2 = GATv2Conv(self.hidden_dim * self.heads, self.hidden_dim, heads=1, dropout=self.dropout, edge_dim=64)
        
        #for projecting back to the same dimension as the query embedding
        self.projection = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, data):
        x, edge_index, edge_type, batch, query = data.x, data.edge_index, data.edge_type, data.batch, data.query

        x0 = x
        q_node = query[batch]
        interaction = x0 * q_node

        #query conditioned gating to ensure only relevant node information is passed
        gate = torch.sigmoid(self.gate_mlp(torch.cat([x0, q_node], dim=-1)))
        x0 = x0 * gate
        x = torch.cat([x0, q_node, interaction], dim=-1)

        edge_attr = self.edge_emb(edge_type)

        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))

        x = self.projection(x)

        #residual from original embeddings so base semantic similarity is preserved
        x = x + x0
        x = F.normalize(x, p=2, dim=-1)

        return x

    def score(self, node_embs, query_emb):
        return (node_embs * query_emb.unsqueeze(0)).sum(dim=-1)