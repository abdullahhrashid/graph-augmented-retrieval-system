from src.utils.config import load_config
from src.data.graph import build_graph

if __name__ == '__main__':
    config = load_config()
    edges_df, node_df = build_graph(config)

    print(f'Nodes: {len(node_df)}')
    print(f'Edges: {len(edges_df)}')
    print(f'Edge type breakdown:')
    for etype, group in edges_df.groupby('edge_type'):
        print(f'  {etype}: {len(group)}')