import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch_geometric.data import Data, Dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SubgraphDataset(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        self.split = split
        self.config = config
        emb_dir = config['paths']['embeddings']
        graph_dir = config['paths']['graph']
        proc_dir = config['paths']['processed']
        self.expansion_hops = config['retrieval']['expansion_hops']

        #load samples
        samples_df = pd.read_parquet(os.path.join(proc_dir, f'{split}_samples.parquet'))
        self.questions = samples_df['question'].tolist()
        self.supporting_ids = samples_df['supporting_chunk_ids'].tolist()

        #load precomputed faiss seeds
        self.seed_indices = np.load(os.path.join(emb_dir, f'{split}_seed_indices.npy'))

        #load query embeddings
        self.query_embs = np.load(os.path.join(emb_dir, f'{split}_query_embeddings.npy'))

        #load document embeddings
        self.doc_embs = np.load(os.path.join(emb_dir, 'chunk_embeddings.npy'), mmap_mode='r')

        #load chunk id array
        self.chunk_ids = np.load(os.path.join(emb_dir, 'chunk_ids.npy'), allow_pickle=True)

        #build chunk id -> integer index lookup
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        logger.info(f'Loading global graph for {split} dataset...')
        edges_df = pd.read_parquet(os.path.join(graph_dir, 'edges.parquet'))

        #map edge type strings to integers
        self.edge_type_map = {'title_mention': 0, 'entity_overlap': 1}

        #build adjacency list, node_idx -> list of (neighbor_idx, edge_type_int)
        self.adj = defaultdict(list)
        for src_idx, dst_idx, etype in zip(
            edges_df['src_idx'].values,
            edges_df['dst_idx'].values,
            edges_df['edge_type'].values,
        ):
            self.adj[src_idx].append((dst_idx, self.edge_type_map[etype]))

        logger.info(f'{split} dataset: {len(self)} samples, '
                    f'{len(self.adj)} nodes with edges, '
                    f'{len(edges_df)} total edges')

    def len(self):
        return len(self.questions)

    def get(self, idx):
        #seed nodes for this query 
        seeds = set(self.seed_indices[idx].tolist())

        #expand to k hop neighbors
        subgraph_nodes = set(seeds)
        frontier = set(seeds)
        for _ in range(self.expansion_hops):
            next_frontier = set()
            for node in frontier:
                for neighbor, _ in self.adj[node]:
                    if neighbor not in subgraph_nodes:
                        next_frontier.add(neighbor)
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier

        #convert to sorted list for consistent ordering
        subgraph_nodes = sorted(subgraph_nodes)

        #local reindexing: global_idx -> local_idx
        global_to_local = {g: l for l, g in enumerate(subgraph_nodes)}

        #extract node features from the global embedding matrix
        x = torch.tensor(
            np.array(self.doc_embs[subgraph_nodes]),
            dtype=torch.float32,
        )

        #extract subgraph edges (only edges where both endpoints are in the subgraph)
        src_list, dst_list, etype_list = [], [], []
        for g_node in subgraph_nodes:
            for neighbor, etype_int in self.adj[g_node]:
                if neighbor in global_to_local:
                    src_list.append(global_to_local[g_node])
                    dst_list.append(global_to_local[neighbor])
                    etype_list.append(etype_int)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_type = torch.tensor(etype_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        #query embedding
        query = torch.tensor(self.query_embs[idx], dtype=torch.float32)

        #labels: 1 if the node is a supporting document for this query, 0 otherwise
        supporting_set = set()
        for sid in self.supporting_ids[idx]:
            if sid in self.id_to_idx:
                supporting_set.add(self.id_to_idx[sid])

        y = torch.zeros(len(subgraph_nodes), dtype=torch.float32)
        for local_idx, global_idx in enumerate(subgraph_nodes):
            if global_idx in supporting_set:
                y[local_idx] = 1.0

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            query=query,
            num_nodes=len(subgraph_nodes),
        )

        return data
