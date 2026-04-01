import os
import numpy as np
import faiss
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def precompute_seeds(config):
    emb_dir = config['paths']['embeddings']
    seed_k = config['retrieval']['seed_k']

    #load faiss index
    index_path = os.path.join(emb_dir, 'faiss.index')
    logger.info(f'Loading FAISS index from {index_path}')
    index = faiss.read_index(index_path)
    logger.info(f'Index contains {index.ntotal} vectors')

    for split in ('train', 'val', 'test'):
        query_path = os.path.join(emb_dir, f'{split}_query_embeddings.npy')
        if not os.path.exists(query_path):
            logger.warning(f'Skipping {split}: {query_path} not found')
            continue

        query_embs = np.load(query_path)
        logger.info(f'Searching {split}: {len(query_embs)} queries, top-{seed_k}')

        #faiss search 
        distances, indices = index.search(query_embs, seed_k)

        out_path = os.path.join(emb_dir, f'{split}_seed_indices.npy')
        np.save(out_path, indices)
        logger.info(f'Saved {split} seeds -> {out_path}  shape={indices.shape}')

        #save distances too, might use them later on
        dist_path = os.path.join(emb_dir, f'{split}_seed_distances.npy')
        np.save(dist_path, distances)

    logger.info('Seed precomputation complete')


if __name__ == '__main__':
    config = load_config()
    precompute_seeds(config)
