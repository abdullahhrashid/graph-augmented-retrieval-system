import os
from collections import defaultdict
from itertools import combinations
import ahocorasick
import pandas as pd
import spacy
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_node_mapping(corpus_df):
    return {cid: idx for idx, cid in enumerate(corpus_df['chunk_id'])}

def build_title_mention_edges(corpus_df, config):
    #hyperlink proxy: title of doc A appears in text of doc B -> directed edge B -> A
    #uses aho corasick multi pattern matching for efficiency
    tc = config['graph']['title_mention']
    case_sensitive = tc['case_sensitive']
    min_title_len = tc['min_title_length']

    titles = corpus_df['title'].tolist()
    texts = corpus_df['text'].tolist()
    doc_ids = corpus_df['chunk_id'].tolist()

    if not case_sensitive:
        search_titles = [t.lower() for t in titles]
        search_texts = [t.lower() for t in texts]
    else:
        search_titles = titles
        search_texts = texts

    #build automaton from all titles long enough to be meaningful
    automaton = ahocorasick.Automaton()
    for title, did in zip(search_titles, doc_ids):
        if len(title) >= min_title_len:
            automaton.add_word(title, (title, did))
    automaton.make_automaton()

    edges = set()
    for source_text, source_id in tqdm(
        zip(search_texts, doc_ids), total=len(doc_ids), desc='title-mention'
    ):
        for _, (matched_title, target_id) in automaton.iter(source_text):
            if source_id != target_id:
                edges.add((source_id, target_id))

    logger.info(f'Title-mention edges: {len(edges)}')
    return edges

def build_entity_overlap_edges(corpus_df, config):
    #extract named entities with spacy, connect documents sharing >=1 entity
    ec = config['graph']['entity_overlap']
    spacy_model = ec['spacy_model']
    entity_types = set(ec['entity_types'])
    min_shared = ec['min_shared_entities']
    max_docs = ec.get('max_docs_per_entity', 50)

    logger.info(f'Loading spaCy model: {spacy_model}')
    nlp = spacy.load(spacy_model, disable=['parser', 'lemmatizer', 'textcat'])

    texts = corpus_df['text'].tolist()
    doc_ids = corpus_df['chunk_id'].tolist()

    logger.info(f'Extracting entities from {len(texts)} documents')
    doc_entities = {}
    for doc, did in tqdm(
        zip(nlp.pipe(texts, batch_size=256, n_process=1), doc_ids),
        total=len(texts), desc='NER'
    ):
        ents = {ent.text.lower().strip() for ent in doc.ents if ent.label_ in entity_types}
        doc_entities[did] = ents

    #inverted index: entity -> set of doc IDs that contain it
    entity_to_docs = defaultdict(set)
    for did, ents in doc_entities.items():
        for ent in ents:
            entity_to_docs[ent].add(did)

    if min_shared == 1:
        #fast path: any shared entity = edge
        edges = set()
        for dids in tqdm(entity_to_docs.values(), desc='entity-overlap edges'):
            did_list = list(dids)
            if len(did_list) < 2 or len(did_list) > max_docs:
                continue
            for a, b in combinations(did_list, 2):
                edges.add((a, b))
                edges.add((b, a))
    else:
        #count shared entities per pair, then threshold
        pair_counts = defaultdict(int)
        for dids in entity_to_docs.values():
            did_list = list(dids)
            if len(did_list) < 2 or len(did_list) > max_docs:
                continue
            for a, b in combinations(did_list, 2):
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1
        edges = {pair for pair, count in pair_counts.items() if count >= min_shared}

    logger.info(f'Entity-overlap edges: {len(edges)}')
    return edges

def build_graph(config):
    proc_dir = config['paths']['processed']
    graph_dir = config['paths']['graph']
    os.makedirs(graph_dir, exist_ok=True)

    corpus_df = pd.read_parquet(os.path.join(proc_dir, 'corpus.parquet'))

    #node mapping: doc_id (chunk_id column) -> integer index
    node_map = build_node_mapping(corpus_df)
    node_df = corpus_df[['chunk_id', 'title']].copy()
    node_df.insert(0, 'node_idx', range(len(node_df)))
    node_df.to_parquet(os.path.join(graph_dir, 'node_mapping.parquet'), index=False)
    logger.info(f'Node mapping: {len(node_df)} nodes')

    all_edge_rows = []
    edge_cfg = config['graph']['edge_types']

    if edge_cfg['title_mention']:
        tm_edges = build_title_mention_edges(corpus_df, config)
        for src, dst in tm_edges:
            all_edge_rows.append({
                'src_id': src, 'dst_id': dst,
                'src_idx': node_map[src], 'dst_idx': node_map[dst],
                'edge_type': 'title_mention',
            })

    if edge_cfg['entity_overlap']:
        eo_edges = build_entity_overlap_edges(corpus_df, config)
        for src, dst in eo_edges:
            all_edge_rows.append({
                'src_id': src, 'dst_id': dst,
                'src_idx': node_map[src], 'dst_idx': node_map[dst],
                'edge_type': 'entity_overlap',
            })

    edges_df = pd.DataFrame(all_edge_rows)
    edges_df = edges_df.drop_duplicates(subset=['src_id', 'dst_id', 'edge_type'])
    edges_df.to_parquet(os.path.join(graph_dir, 'edges.parquet'), index=False)

    logger.info(f'Total edges: {len(edges_df)}')
    for etype, group in edges_df.groupby('edge_type'):
        logger.info(f'  {etype}: {len(group)} edges')

    logger.info('Graph build complete')
    return edges_df, node_df
