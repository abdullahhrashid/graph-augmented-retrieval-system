import os
import random
import time
import numpy as np
import pandas as pd
import torch
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.models.gnn import GraphRanker
from src.utils.logger import get_logger
from torch_geometric.data import Data

logger = get_logger(__name__)

load_dotenv()

class RetrievalPipeline:
    def __init__(self, config):
        self.config = config
        self.top_k_default = config['serving']['top_k']
        self.seed_k = config['retrieval']['seed_k']
        self.expansion_hops = config['retrieval']['expansion_hops']
        self.max_neighbors = config['retrieval'].get('max_neighbors_per_hop', 40)

        emb_dir = config['paths']['embeddings']
        graph_dir = config['paths']['graph']
        proc_dir = config['paths']['processed']

        #load embedding model
        model_name = config['embeddings']['model_name']
        logger.info(f'Loading embedding model: {model_name}')
        self.encoder = SentenceTransformer(model_name)

        #load faiss index
        index_path = os.path.join(emb_dir, 'faiss.index')
        logger.info(f'Loading FAISS index: {index_path}')
        self.faiss_index = faiss.read_index(index_path)

        #load document embeddings and IDs
        logger.info('Loading document embeddings...')
        self.doc_embs = np.load(os.path.join(emb_dir, 'chunk_embeddings.npy'))
        self.chunk_ids = np.load(os.path.join(emb_dir, 'chunk_ids.npy'), allow_pickle=True)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        #load corpus texts for response
        logger.info('Loading corpus texts...')
        corpus_df = pd.read_parquet(os.path.join(proc_dir, 'corpus.parquet'))
        self.chunk_texts = {row['chunk_id']: row['text'] for _, row in corpus_df.iterrows()}

        #load graph adjacency
        logger.info('Loading graph...')
        edges_df = pd.read_parquet(os.path.join(graph_dir, 'edges.parquet'))
        edge_type_map = {'title_mention': 0, 'entity_overlap': 1}
        self.adj = defaultdict(list)
        for src_idx, dst_idx, etype in zip(
            edges_df['src_idx'].values,
            edges_df['dst_idx'].values,
            edges_df['edge_type'].values,
        ):
            self.adj[src_idx].append((dst_idx, edge_type_map[etype]))

        #load gnn model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = config['serving']['checkpoint']
        logger.info(f'Loading GNN model from {checkpoint_path}')
        self.model = GraphRanker(config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

        logger.info(f'Pipeline ready on {self.device} | '
                     f'{self.faiss_index.ntotal} docs | '
                     f'{len(self.adj)} graph nodes')

    def expand_subgraph(self, seeds):
        subgraph_nodes = set(seeds)
        frontier = set(seeds)
        for _ in range(self.expansion_hops):
            next_frontier = set()
            for node in frontier:
                neighbors = self.adj[node]
                if len(neighbors) > self.max_neighbors:
                    neighbors = random.sample(neighbors, self.max_neighbors)
                for neighbor, _ in neighbors:
                    if neighbor not in subgraph_nodes:
                        next_frontier.add(neighbor)
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier
        return subgraph_nodes

    def build_pyg_data(self, subgraph_nodes, query_emb):
        node_list = sorted(subgraph_nodes)
        global_to_local = {g: l for l, g in enumerate(node_list)}

        x = torch.tensor(self.doc_embs[node_list], dtype=torch.float32)

        src_list, dst_list, etype_list = [], [], []
        for g_node in node_list:
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

        query = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            query=query,
            num_nodes=len(node_list),
        )
        data.batch = torch.zeros(len(node_list), dtype=torch.long)
        return data, node_list

    def retrieve(self, query: str, top_k: int = None) -> dict:
        if top_k is None:
            top_k = self.top_k_default

        t0 = time.time()

        #embed query
        query_emb = self.encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        #faiss retrieval - get seed documents
        distances, indices = self.faiss_index.search(
            query_emb.reshape(1, -1), self.seed_k
        )
        seed_indices = set(indices[0].tolist())

        #bfs expansion
        subgraph_nodes = self.expand_subgraph(seed_indices)

        #gnn re-ranking
        pyg_data, node_list = self.build_pyg_data(subgraph_nodes, query_emb)
        pyg_data = pyg_data.to(self.device)

        with torch.no_grad():
            refined = self.model(pyg_data)
            q_emb = pyg_data.query.to(self.device)
            scores = (refined * q_emb).sum(dim=-1).cpu().numpy()

        #rank and collect results
        top_local = np.argsort(-scores)[:top_k]

        results = []
        for rank, local_idx in enumerate(top_local):
            global_idx = node_list[local_idx]
            chunk_id = self.chunk_ids[global_idx]
            results.append({
                'rank': rank + 1,
                'chunk_id': str(chunk_id),
                'score': round(float(scores[local_idx]), 4),
                'text': self.chunk_texts.get(chunk_id, ''),
            })

        latency_ms = (time.time() - t0) * 1000

        return {
            'query': query,
            'top_k': top_k,
            'results': results,
            'metadata': {
                'model': 'GATv2Conv-GraphRanker',
                'subgraph_nodes': len(node_list),
                'subgraph_edges': pyg_data.edge_index.size(1),
            },
        }
