import hashlib
import os
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)

def make_doc_id(title):
    return hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]

def extract_corpus_and_samples(config):
    raw_path = config['paths']['raw_data']
    min_len = config['corpus']['min_doc_length']  
    corpus = {}          
    split_dfs = {}

    for split in ('train', 'val', 'test'):
        split_path = os.path.join(raw_path, split)
        logger.info(f'Processing split: {split} from {split_path}')
        ds = load_from_disk(split_path)

        rows = []
        for idx, sample in enumerate(tqdm(ds, desc=f'{split}')):
            titles = sample['context']['title']
            sentences_list = sample['context']['sentences']

            #supporting facts for this sample
            sf_titles = sample['supporting_facts']['title']
            sf_sent_ids = sample['supporting_facts']['sent_id']
            #group supporting facts by title and set of sentence indices
            sf_map = {}
            for t, s in zip(sf_titles, sf_sent_ids):
                sf_map.setdefault(t, set()).add(s)

            context_doc_ids = []
            supporting_doc_ids = []
            supporting_sent_ids_per_doc = {}

            for title, sents in zip(titles, sentences_list):
                text = ' '.join(sents).strip()
                if len(text) < min_len:
                    continue

                doc_id = make_doc_id(title)

                #add to corpus (dedup by title — same article always has same text)
                if title not in corpus:
                    corpus[title] = {'chunk_id': doc_id, 'text': text}

                context_doc_ids.append(doc_id)

                #check if this document is a supporting fact for this sample
                if title in sf_map:
                    supporting_doc_ids.append(doc_id)
                    supporting_sent_ids_per_doc[doc_id] = sorted(sf_map[title])

            rows.append({
                'sample_id': sample['id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'context_chunk_ids': context_doc_ids,
                'supporting_chunk_ids': supporting_doc_ids,
            })

        split_dfs[split] = pd.DataFrame(rows)
        logger.info(f'{split}: {len(rows)} samples processed')

    #build corpus dataframe
    corpus_rows = [
        {'chunk_id': v['chunk_id'], 'title': title, 'text': v['text']}
        for title, v in corpus.items()
    ]
    corpus_df = pd.DataFrame(corpus_rows)
    logger.info(f'Unified corpus: {len(corpus_df)} unique documents')

    return corpus_df, split_dfs

def save_corpus_and_samples(corpus_df, split_dfs, config):
    out_dir = config['paths']['processed']
    os.makedirs(out_dir, exist_ok=True)

    corpus_path = os.path.join(out_dir, 'corpus.parquet')
    corpus_df.to_parquet(corpus_path, index=False)
    logger.info(f'Saved corpus -> {corpus_path}  ({len(corpus_df)} documents)')

    for split, df in split_dfs.items():
        path = os.path.join(out_dir, f'{split}_samples.parquet')
        df.to_parquet(path, index=False)
        logger.info(f'Saved {split} samples -> {path}  ({len(df)} rows)')


def build_corpus(config):
    corpus_df, split_dfs = extract_corpus_and_samples(config)
    save_corpus_and_samples(corpus_df, split_dfs, config)
    return corpus_df, split_dfs
