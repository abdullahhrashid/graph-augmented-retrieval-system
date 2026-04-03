import argparse
import os
import time
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from src.utils.config import load_config
from src.utils.logger import get_logger
from dotenv import load_dotenv
from src.data.dataset import SubgraphDataset
from src.models.gnn import GraphRanker
from src.models.loss import MultiPositiveSigmoidLoss

load_dotenv()

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GraphRanker GNN')
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to config file')
    parser.add_argument('--run_name', type=str, required=True, help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    return parser.parse_args()

def get_device(config):
    device_str = config['training'].get('device')
    if device_str:
        return torch.device(device_str)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_loss_inputs(data, refined_embs):
    query_embs = data.query            # [Q, D]
    doc_embs = refined_embs            # [N, D]
    
    Q = query_embs.size(0)
    N = doc_embs.size(0)
    labels = torch.zeros(Q, N, device=refined_embs.device)
    
    for g in range(Q):
        #mark positive docs for graph g
        pos_mask = (data.batch == g) & (data.y == 1.0)
        labels[g, pos_mask] = 1.0
        
    return query_embs, doc_embs, labels

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_pos = 0
    total_pos_found = 0

    for data in tqdm(val_loader, desc='Validating', leave=False):
        data = data.to(device)
        refined = model(data)

        query_embs, doc_embs, labels = build_loss_inputs(data, refined)
        if query_embs is None:
            continue

        loss = criterion(query_embs, doc_embs, labels)
        total_loss += loss.item()
        num_batches += 1

        #compute recall@10 as a quick metric
        batch_vec = data.batch
        num_graphs = batch_vec.max().item() + 1
        for g in range(num_graphs):
            node_mask = (batch_vec == g)
            pos_mask = node_mask & (data.y == 1.0)
            num_pos = pos_mask.sum().item()
            if num_pos == 0:
                continue

            #score all nodes in this subgraph against the query
            node_embs = refined[node_mask]
            q = data.query[g]
            scores = (node_embs * q.unsqueeze(0)).sum(dim=-1)

            #top 10 by score
            k = min(10, scores.size(0))
            _, top_indices = scores.topk(k)

            #how many positives are in the top 10
            local_pos_mask = data.y[node_mask]
            retrieved_labels = local_pos_mask[top_indices]
            total_pos_found += retrieved_labels.sum().item()
            total_pos += num_pos

    avg_loss = total_loss / max(num_batches, 1)
    recall_at_10 = total_pos_found / max(total_pos, 1)

    return avg_loss, recall_at_10

def train(args, config):
    device = get_device(config)
    logger.info(f'Using device: {device}')

    #read hyperparams
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    patience = config['training']['patience']
    warmup_epochs = config['training']['warmup_epochs']
    min_lr = config['training']['min_lr']
    wandb_project = config['training']['wandb_project']
    save_dir = config['training']['save_dir']
    if not os.path.isabs(save_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        save_dir = os.path.join(project_root, save_dir)
    num_workers = config['training']['num_workers']

    #wandb init
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=args.run_name,
                config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'patience': patience,
                    'warmup_epochs': warmup_epochs,
                    'min_lr': min_lr,
                    'hidden_dim': config['gnn']['hidden_dim'],
                    'heads': config['gnn']['heads'],
                    'dropout': config['gnn']['dropout'],
                    'seed_k': config['retrieval']['seed_k'],
                    'expansion_hops': config['retrieval']['expansion_hops'],
                },
            )
            logger.info(f'Wandb run: {wandb_run.url}')
        except ImportError:
            logger.warning('wandb not installed, disabling tracking')
            wandb_run = None

    #datasets
    logger.info('Loading train dataset...')
    train_dataset = SubgraphDataset(config, split='train')
    logger.info('Loading val dataset...')
    val_dataset = SubgraphDataset(config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #model, loss, optimizer, scheduler
    model = GraphRanker(config).to(device)
    criterion = MultiPositiveSigmoidLoss().to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params:,}')

    #checkpointing
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    #training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        skipped = 0
        t0 = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{epochs}')
        for batch_idx, data in pbar:
            data = data.to(device)

            refined = model(data)
            query_embs, doc_embs, labels = build_loss_inputs(data, refined)

            if query_embs is None:
                skipped += 1
                continue

            loss = criterion(query_embs, doc_embs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg = epoch_loss / num_batches
                pbar.set_postfix(loss=f'{avg:.4f}', skipped=skipped)

        #lr scheduling
        if epoch > warmup_epochs:
            scheduler.step()

        train_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        #validation
        val_loss, val_recall = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f'Epoch {epoch}/{epochs} | '
            f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
            f'val_R@10={val_recall:.4f} | lr={current_lr:.2e} | '
            f'time={elapsed:.1f}s | skipped={skipped}'
        )

        #wandb logging
        if wandb_run:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_recall@10': val_recall,
                'lr': current_lr,
                'epoch_time_s': elapsed,
                'skipped_batches': skipped
            })

        #early stopping + checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recall@10': val_recall,
                'config': config,
            }, ckpt_path)
            logger.info(f'  Saved best model -> {ckpt_path}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f'Early stopping at epoch {epoch} (patience={patience})')
                break

    #save final model
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    logger.info(f'Saved final model -> {final_path}')

    if wandb_run:
        wandb.finish()

    logger.info('Training complete!')


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(args, config)
