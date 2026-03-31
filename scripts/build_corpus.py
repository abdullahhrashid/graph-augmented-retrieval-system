from src.utils.config import load_config
from src.data.corpus import build_corpus

if __name__ == '__main__':
    config = load_config()
    corpus_df, split_dfs = build_corpus(config)

    print(f'Corpus: {len(corpus_df)} unique documents')
    for split, df in split_dfs.items():
        n_support = df['supporting_chunk_ids'].apply(len).sum()
        print(f'{split}: {len(df)} samples, {n_support} total supporting facts')
