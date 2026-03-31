import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def encode_texts(model, texts, batch_size, normalize):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)

def build_vectorstore(config):
    proc_dir = config['paths']['processed']
    emb_dir = config['paths']['embeddings']
    os.makedirs(emb_dir, exist_ok=True)

    emb_cfg = config['embeddings']
    model_name = emb_cfg['model_name']
    batch_size = emb_cfg['batch_size']
    normalize = emb_cfg['normalize']

    logger.info(f'Loading embedding model: {model_name}')
    model = SentenceTransformer(model_name)

    #encode chunks
    corpus_df = pd.read_parquet(os.path.join(proc_dir, 'corpus.parquet'))
    chunk_ids = corpus_df['chunk_id'].tolist()
    chunk_texts = corpus_df['text'].tolist()

    logger.info(f'Encoding {len(chunk_texts)} chunks')
    chunk_embs = encode_texts(model, chunk_texts, batch_size, normalize)

    np.save(os.path.join(emb_dir, 'chunk_embeddings.npy'), chunk_embs)
    np.save(os.path.join(emb_dir, 'chunk_ids.npy'), np.array(chunk_ids))
    logger.info(f'Saved chunk embeddings: {chunk_embs.shape}')

    #build faiss index
    dim = chunk_embs.shape[1]
    index_type = config['faiss']['index_type']

    if index_type == 'flat':
        index = faiss.IndexFlatIP(dim)
    elif index_type == 'ivf':
        nlist = min(256, len(chunk_ids) // 10)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(chunk_embs)
        index.nprobe = config['faiss']['nprobe']
    else:
        raise ValueError(f'Unknown index type: {index_type}')

    index.add(chunk_embs)
    index_path = os.path.join(emb_dir, 'faiss.index')
    faiss.write_index(index, index_path)
    logger.info(f'FAISS index ({index_type}): {index.ntotal} vectors -> {index_path}')

    #encode queries per split
    for split in ('train', 'val', 'test'):
        samples_path = os.path.join(proc_dir, f'{split}_samples.parquet')
        if not os.path.exists(samples_path):
            logger.warning(f'Skipping {split}: {samples_path} not found')
            continue

        df = pd.read_parquet(samples_path)
        questions = df['question'].tolist()

        logger.info(f'Encoding {len(questions)} {split} queries')
        query_embs = encode_texts(model, questions, batch_size, normalize)

        np.save(os.path.join(emb_dir, f'{split}_query_embeddings.npy'), query_embs)
        logger.info(f'Saved {split} query embeddings: {query_embs.shape}')

    logger.info('Vector store build complete')
