import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.gnn import GraphRanker
from src.evaluation.evaluator import evaluate_system

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate retrieval systems')
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to GNN checkpoint')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20])
    return parser.parse_args()

def load_shared_data(config, split):
    emb_dir = config['paths']['embeddings']
    graph_dir = config['paths']['graph']
    proc_dir = config['paths']['processed']

    #samples with gold doc IDs
    samples_df = pd.read_parquet(os.path.join(proc_dir, f'{split}_samples.parquet'))
    all_gold_ids = samples_df['supporting_chunk_ids'].tolist()

    #precomputed faiss seeds and distances
    seed_indices = np.load(os.path.join(emb_dir, f'{split}_seed_indices.npy'))
    seed_distances = np.load(os.path.join(emb_dir, f'{split}_seed_distances.npy'))

    #query and doc embeddings
    query_embs = np.load(os.path.join(emb_dir, f'{split}_query_embeddings.npy'))
    doc_embs = np.load(os.path.join(emb_dir, 'chunk_embeddings.npy'), mmap_mode='r')

    chunk_ids = np.load(os.path.join(emb_dir, 'chunk_ids.npy'), allow_pickle=True)

    #graph adjacency list
    edges_df = pd.read_parquet(os.path.join(graph_dir, 'edges.parquet'))
    edge_type_map = {'title_mention': 0, 'entity_overlap': 1}
    adj = defaultdict(list)
    for src_idx, dst_idx, etype in zip(
        edges_df['src_idx'].values,
        edges_df['dst_idx'].values,
        edges_df['edge_type'].values,
    ):
        adj[src_idx].append((dst_idx, edge_type_map[etype]))

    #id_to_idx lookup
    id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}

    return {
        'all_gold_ids': all_gold_ids,
        'seed_indices': seed_indices,
        'seed_distances': seed_distances,
        'query_embs': query_embs,
        'doc_embs': doc_embs,
        'chunk_ids': chunk_ids,
        'adj': adj,
        'id_to_idx': id_to_idx,
    }

def run_vector_rag(data, k_values):
    logger.info('Running Vector RAG')
    all_ranked = []
    max_k = max(k_values)

    for i in range(len(data['seed_indices'])):
        #seeds are already ranked by faiss distance (descending similarity)
        top_indices = data['seed_indices'][i][:max_k]
        ranked_ids = [data['chunk_ids'][idx] for idx in top_indices]
        all_ranked.append(ranked_ids)

    return all_ranked

def expand_subgraph(seeds, adj, expansion_hops, max_neighbors):
    subgraph_nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(expansion_hops):
        next_frontier = set()
        for node in frontier:
            neighbors = adj[node]
            if len(neighbors) > max_neighbors:
                neighbors = random.sample(neighbors, max_neighbors)
            for neighbor, _ in neighbors:
                if neighbor not in subgraph_nodes:
                    next_frontier.add(neighbor)
        subgraph_nodes.update(next_frontier)
        frontier = next_frontier
    return subgraph_nodes


def run_graph_rag(data, config, k_values):
    logger.info('Running Graph RAG')
    all_ranked = []
    max_k = max(k_values)
    expansion_hops = config['retrieval']['expansion_hops']
    max_neighbors = config['retrieval'].get('max_neighbors_per_hop', 40)

    for i in tqdm(range(len(data['seed_indices'])), desc='Graph RAG'):
        seeds = set(data['seed_indices'][i].tolist())
        subgraph_nodes = expand_subgraph(seeds, data['adj'], expansion_hops, max_neighbors)

        #score all nodes by cosine similarity with query
        node_list = sorted(subgraph_nodes)
        node_embs = np.array(data['doc_embs'][node_list])  
        query = data['query_embs'][i]  # [D]

        scores = node_embs @ query  # cosine sim (embeddings are normalized)

        #rank by score descending
        top_local = np.argsort(-scores)[:max_k]
        ranked_ids = [data['chunk_ids'][node_list[j]] for j in top_local]
        all_ranked.append(ranked_ids)

    return all_ranked

def run_gnn_rag(data, config, checkpoint_path, k_values):
    logger.info('Running GNN RAG')
    from torch_geometric.data import Data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expansion_hops = config['retrieval']['expansion_hops']
    max_neighbors = config['retrieval'].get('max_neighbors_per_hop', 40)
    max_k = max(k_values)

    #load model
    model = GraphRanker(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    logger.info(f'Loaded GNN from {checkpoint_path}')

    all_ranked = []

    for i in tqdm(range(len(data['seed_indices'])), desc='GNN RAG'):
        seeds = set(data['seed_indices'][i].tolist())
        subgraph_nodes_set = expand_subgraph(seeds, data['adj'], expansion_hops, max_neighbors)
        subgraph_nodes = sorted(subgraph_nodes_set)
        global_to_local = {g: l for l, g in enumerate(subgraph_nodes)}

        #node features
        x = torch.tensor(np.array(data['doc_embs'][subgraph_nodes]), dtype=torch.float32)

        #edges
        src_list, dst_list, etype_list = [], [], []
        for g_node in subgraph_nodes:
            for neighbor, etype_int in data['adj'][g_node]:
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

        query = torch.tensor(data['query_embs'][i], dtype=torch.float32).unsqueeze(0)

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.zeros(len(subgraph_nodes)),
            query=query,
            num_nodes=len(subgraph_nodes),
        )
        pyg_data.batch = torch.zeros(len(subgraph_nodes), dtype=torch.long)
        pyg_data = pyg_data.to(device)

        with torch.no_grad():
            refined = model(pyg_data)
            q_emb = query.to(device)
            scores = (refined * q_emb).sum(dim=-1)

        scores_np = scores.cpu().numpy()
        top_local = np.argsort(-scores_np)[:max_k]
        ranked_ids = [data['chunk_ids'][subgraph_nodes[j]] for j in top_local]
        all_ranked.append(ranked_ids)

    return all_ranked

def print_results_table(results_dict, k_values):
    systems = list(results_dict.keys())

    #build metric keys in display order
    metric_keys = []
    for k in k_values:
        metric_keys.extend([f'Recall@{k}', f'EM@{k}', f'nDCG@{k}'])
    metric_keys.append('MAP')

    #header
    col_width = 14
    header = f'{"Metric":<{col_width}}'
    for sys_name in systems:
        header += f'{sys_name:>{col_width}}'
    print('=' * len(header))
    print(header)
    print('-' * len(header))

    for key in metric_keys:
        row = f'{key:<{col_width}}'
        for sys_name in systems:
            val = results_dict[sys_name].get(key, 0.0)
            row += f'{val:>{col_width}.4f}'
        print(row)

    print('=' * len(header))


def main():
    args = parse_args()
    config = load_config(args.config)
    k_values = args.k_values

    logger.info(f'Evaluating on {args.split} set with K={k_values}')

    #load shared data
    data = load_shared_data(config, args.split)
    n_queries = len(data['all_gold_ids'])
    logger.info(f'{n_queries} queries loaded')

    results = {}

    #vector rag
    vec_ranked = run_vector_rag(data, k_values)
    results['Vector RAG'] = evaluate_system(vec_ranked, data['all_gold_ids'], k_values)

    #graph rag
    graph_ranked = run_graph_rag(data, config, k_values)
    results['Graph RAG'] = evaluate_system(graph_ranked, data['all_gold_ids'], k_values)

    #graph rag with gnn
    gnn_ranked = run_gnn_rag(data, config, args.checkpoint, k_values)
    results['GNN RAG'] = evaluate_system(gnn_ranked, data['all_gold_ids'], k_values)

    #print table
    print('\n')
    print_results_table(results, k_values)

    #save results to csv
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for sys_name, metrics in results.items():
        row = {'System': sys_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f'{args.split}_metrics.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved results -> {csv_path}')

if __name__ == '__main__':
    main()
