import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset folder (without extension).")
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=float, default=0.1, help='KG loss weight')
parser.add_argument('--margin', type=float, default=1.0)
args = parser.parse_args()

inter_df = pd.read_csv(f"{args.dataset_path}.inter", sep='\t', dtype=str)
kg_df = pd.read_csv(f"{args.dataset_path}.kg", sep='\t', dtype=str)
link_df = pd.read_csv(f"{args.dataset_path}.link", sep='\t', dtype=str)

if 'user_id:token' not in inter_df.columns:
    inter_df.columns = ['user_id:token','item_id:token','rating:float'] if inter_df.shape[1] >= 3 else ['user_id:token','item_id:token']
if 'head_id:token' not in kg_df.columns:
    kg_df.columns = ['head_id:token','relation_id:token','tail_id:token']
if 'item_id:token' not in link_df.columns:
    link_df.columns = ['item_id:token','entity_id:token']

if 'rating:float' not in inter_df.columns:
    inter_df['rating:float'] = '1.0'
if 'timestamp:float' not in inter_df.columns:
    inter_df['timestamp:float'] = '0'

users = inter_df['user_id:token'].unique().tolist()
items_tokens = inter_df['item_id:token'].unique().tolist()
kg_entities_tokens = pd.concat([kg_df['head_id:token'], kg_df['tail_id:token']]).unique().tolist()
entities_tokens = list(dict.fromkeys(kg_entities_tokens + items_tokens + users))

relations_tokens = list(kg_df['relation_id:token'].unique().tolist())
entity2id = {tok: idx+1 for idx, tok in enumerate(entities_tokens)}
relation2id = {tok: idx+1 for idx, tok in enumerate(relations_tokens)}
if 'interact' not in relation2id:
    relation2id['interact'] = len(relation2id)+1

DEFAULT_ENTITY_ID = 0

item2id = {tok: idx+1 for idx, tok in enumerate(items_tokens)}
id2item = {v:k for k,v in item2id.items()}

num_items = len(item2id)
num_entities = max(entity2id.values()) if len(entity2id)>0 else 0
num_entities = num_entities + 1
num_relations = max(relation2id.values()) if len(relation2id)>0 else 0
num_relations = num_relations + 1


item2entity_token = link_df.set_index('item_id:token')['entity_id:token'].to_dict()
item2entity = {it: int(entity2id.get(item2entity_token.get(it, None), DEFAULT_ENTITY_ID)) for it in items_tokens}

kg_triplets = []
for h_t, r_t, t_t in zip(kg_df['head_id:token'], kg_df['relation_id:token'], kg_df['tail_id:token']):
    h_id = entity2id.get(h_t, DEFAULT_ENTITY_ID)
    r_id = relation2id.get(r_t, 0)
    t_id = entity2id.get(t_t, DEFAULT_ENTITY_ID)
    kg_triplets.append((int(h_id), int(r_id), int(t_id)))

user2seq_tokens = inter_df.groupby('user_id:token')['item_id:token'].apply(list).to_dict()
user2seq = {}
for u, seq_tokens in user2seq_tokens.items():
    seq_ids = [item2id.get(t, 0) for t in seq_tokens]
    seq_ids = [s for s in seq_ids if s != 0]
    if len(seq_ids) >= 1:
        user2seq[u] = seq_ids

max_seq_len = max(len(s) for s in user2seq.values())


class SessionDataset(Dataset):
    def __init__(self, user2seq, item2id, max_len):
        self.user_list = list(user2seq.keys())
        self.seqs = [user2seq[u] for u in self.user_list]
        self.item2id = item2id
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        if seq_len < self.max_len:
            padded = [0] * (self.max_len - seq_len) + seq
        else:
            padded = seq
        pos_item = seq[-1]
        return {
            'seq': torch.LongTensor(padded),
            'seq_len': torch.LongTensor([seq_len]),
            'pos_item': torch.LongTensor([pos_item]),
            'user': self.user_list[idx]
        }

dataset = SessionDataset(user2seq, item2id, max_seq_len)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, emb_dim, padding_idx=0)
        self.rel_emb = nn.Embedding(num_relations, emb_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def forward(self, h, r, t):
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        return h_e, r_e, t_e

    def score(self, h_e, r_e, t_e):
        return -torch.norm(h_e + r_e - t_e, p=2, dim=1)

class GNN(nn.Module):
    def __init__(self, embedding_size, step=1):
        super().__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden))
        input_out = torch.matmul(A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden))
        inputs = torch.cat([input_in, input_out], 2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        out = self.encoder(x, src_key_padding_mask=mask)
        out = self.layer_norm(out)
        return out

class GCSAN_KG_Model(nn.Module):
    def __init__(self, n_items, n_entities, n_relations, emb_dim=64, gnn_step=1, trans_layers=2, trans_heads=2, reg_weight=1e-4):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, emb_dim, padding_idx=0)
        self.entity_emb = nn.Embedding(n_entities, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(n_relations, emb_dim, padding_idx=0)

        self.kg_model = TransE(n_entities, n_relations, emb_dim)

        self.gnn = GNN(emb_dim, step=gnn_step)
        self.transformer = SimpleTransformerEncoder(n_layers=trans_layers, n_heads=trans_heads, hidden_size=emb_dim)
        self.reg_weight = reg_weight
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, emb_dim)

    def _get_slice(self, item_seq):
        item_seq_np = item_seq.cpu().numpy()
        batch_size = item_seq.shape[0]
        max_n_node = item_seq.shape[1]
        items_list = []
        A_list = []
        alias_inputs = []
        for u_input in item_seq_np:
            node = np.unique(u_input)
            items = node.tolist() + (max_n_node - len(node)) * [0]
            items_list.append(items)
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A_comb = np.concatenate([u_A_in, u_A_out]).transpose()
            A_list.append(u_A_comb)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        alias_inputs = torch.LongTensor(alias_inputs).to(item_seq.device)
        A = torch.FloatTensor(np.array(A_list)).to(item_seq.device)
        items = torch.LongTensor(np.array(items_list)).to(item_seq.device)
        return alias_inputs, A, items

    def forward(self, item_seq):
        alias_inputs, A, items = self._get_slice(item_seq)
        hidden = self.item_emb(items)
        hidden = self.gnn(A, hidden)
        alias_inputs_exp = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.emb_dim)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs_exp)
        out = self.transformer(seq_hidden)
        seq_output = out[:, -1, :]
        seq_output = self.fc(seq_output)
        return seq_output

    def transE_score(self, h, r, t):
        h_e, r_e, t_e = self.kg_model(h, r, t)
        return self.kg_model.score(h_e, r_e, t_e)

    def reg_loss(self):
        return torch.mean(self.item_emb.weight.pow(2))

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores + 1e-12))

def transE_margin_loss(model, triplets, num_entities, margin):
    if len(triplets) == 0:
        return torch.tensor(0.0, device=device)
    h, r, t = zip(*random.sample(triplets, min(len(triplets), 256)))
    h = torch.tensor(h, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.long, device=device)
    t = torch.tensor(t, dtype=torch.long, device=device)
    t_neg = torch.randint(0, num_entities, t.shape, device=device)
    pos_h_e, pos_r_e, pos_t_e = model.kg_model(h, r, t)
    neg_h_e, neg_r_e, neg_t_e = model.kg_model(h, r, t_neg)
    pos_score = -torch.norm(pos_h_e + pos_r_e - pos_t_e, p=2, dim=1)
    neg_score = -torch.norm(neg_h_e + neg_r_e - neg_t_e, p=2, dim=1)
    loss = torch.mean(F.relu(margin + neg_score - pos_score))
    return loss

def recall_at_k(ranklist, gt, k):
    return 1.0 if gt in ranklist[:k] else 0.0

def ndcg_at_k(ranklist, gt, k):
    if gt in ranklist[:k]:
        idx = ranklist.index(gt)
        return 1.0 / math.log2(idx + 2)
    return 0.0

def evaluate(model, dataloader, K_list=[20,50], device=device):
    model.eval()
    recalls = {k: [] for k in K_list}
    ndcgs = {k: [] for k in K_list}

    with torch.no_grad():
        for batch in dataloader:
            seqs = batch['seq'].to(device)
            seq_lens = batch['seq_len'].squeeze(1).to(device)
            pos_items = batch['pos_item'].squeeze(1).to(device)

            seq_repr = model.forward(seqs)
            item_embs = model.item_emb.weight
            scores = torch.matmul(seq_repr, item_embs.T)
            for i in range(seqs.size(0)):
                seen = set(seqs[i].cpu().tolist())
                gt = int(pos_items[i].item())
                if gt == 0:
                    continue
                for it in seen:
                    if it != 0 and it != gt:
                        scores[i, it] = -1e9

            maxk = max(K_list)
            _, topk_idx = torch.topk(scores, maxk, dim=1)
            topk_idx = topk_idx.cpu().tolist()

            for i, ranklist in enumerate(topk_idx):
                gt = int(pos_items[i].item())
                if gt == 0:
                    continue
                for k in K_list:
                    recalls[k].append(1.0 if gt in ranklist[:k] else 0.0)
                    if gt in ranklist[:k]:
                        idx = ranklist.index(gt)
                        ndcgs[k].append(1.0 / math.log2(idx + 2))
                    else:
                        ndcgs[k].append(0.0)

    results = {f'Recall@{k}': float(np.mean(recalls[k])) for k in K_list}
    results.update({f'NDCG@{k}': float(np.mean(ndcgs[k])) for k in K_list})
    return results


model = GCSAN_KG_Model(num_items, num_entities, num_relations, emb_dim=args.emb_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
bpr = BPRLoss()

with torch.no_grad():
    for token, iid in item2id.items():
        idx = iid
        ent_tok = item2entity_token.get(token, None)
        ent_id = entity2id.get(ent_tok, None)
        if ent_id is not None:
            model.item_emb.weight[idx] = model.kg_model.ent_emb.weight[ent_id]

for epoch in range(1, args.epochs+1):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        seqs = batch['seq'].to(device)
        seq_lens = batch['seq_len'].squeeze(1).to(device).long()
        pos_items = batch['pos_item'].squeeze(1).to(device)
        seq_out = model.forward(seqs)
        neg_items = torch.randint(1, num_items+1, pos_items.shape, device=device)
        pos_emb = model.item_emb(pos_items)
        neg_emb = model.item_emb(neg_items)
        pos_scores = torch.sum(seq_out * pos_emb, dim=1)
        neg_scores = torch.sum(seq_out * neg_emb, dim=1)
        loss_bpr = bpr(pos_scores, neg_scores)

        loss_kge = transE_margin_loss(model, kg_triplets, num_entities, args.margin)

        loss = loss_bpr + args.alpha * loss_kge + model.reg_weight * model.reg_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

    metrics = evaluate(model, train_loader, K_list=[20,50])
    print(f"Epoch {epoch} finished. AvgLoss={total_loss/len(train_loader):.6f}")
    print(f"Recall@20={metrics['Recall@20']:.4f}, NDCG@20={metrics['NDCG@20']:.4f}, Recall@50={metrics['Recall@50']:.4f}, NDCG@50={metrics['NDCG@50']:.4f}")


final_item_embedding = model.item_emb.weight.data.cpu().numpy()
final_entity_embedding = model.kg_model.ent_emb.weight.data.cpu().numpy()

print("Done. final_item_embedding.shape=", final_item_embedding.shape, "final_entity_embedding.shape=", final_entity_embedding.shape)
