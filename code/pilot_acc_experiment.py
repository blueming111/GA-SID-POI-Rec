"""
下游推荐指标评估：Acc@1 / Acc@5 / Acc@10（对应原论文的下游LLM推荐指标）

原论文管线：RQVAE学习Semantic ID → LLM微调 → 自回归生成下一个POI的SID → 命中即正确(Acc@1)
本机无transformers/LLaMA-Factory环境，采用轻量代理协议（预研性质实验）：
  用相同的SID序列模型（GRU，结构与训练超参完全一致）分别在两个模型生成的SID上训练，
  预测用户下一个访问的POI。SID质量（地理感知）的差异将直接体现在Acc@1上。
  协议与原论文一致的部分：
    - 输入为离散SID码（3级残差VQ索引），而非连续向量 → 保留碰撞/共享码结构
    - 相邻POI共享码前缀 → 序列模型对邻近POI的泛化更好（语义ID核心收益）
  协议的简化部分：
    - 用GRU替代LLaMA（同样的自回归next-token目标），用dot-product检索替代beam search解码

评估指标：
  Acc@1  命中率@1（原论文主指标）
  Acc@5 / Acc@10 命中率@5/@10（原论文辅助指标）
"""
import os
import sys
import csv
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))

from RQVAE.rqvae import RQVAE as BaselineRQVAE
from gad_rqvae import GADRQVAE
from POIdataset_geo import GeoEmbDataset
from pilot_experiment import build_subset_data


# ============================================================
#  第一步：从原始TSMC2014构建用户签到序列（与build_subset_data完全一致的选择逻辑）
# ============================================================
def build_user_sequences(tsmc_path, num_pois=600):
    """按与build_subset_data相同的规则选top-N POI，构建每个用户的按时间排序的POI访问序列"""
    df = pd.read_csv(tsmc_path, sep='\t', header=None,
                     names=['UserID', 'VenueID', 'CatID', 'CatName', 'Lat', 'Lon', 'TZ', 'Time'])
    df['UserID'] = df['UserID'].astype(str)

    # 与build_subset_data一致：按访问次数降序取前num_pois个POI
    counts = df.groupby('VenueID').size().reset_index(name='Count')
    counts = counts.sort_values('Count', ascending=False).head(num_pois).reset_index(drop=True)
    selected_venues = counts['VenueID'].tolist()
    venue2pid = {v: i for i, v in enumerate(selected_venues)}
    print(f"[序列构建] 选取POI数: {len(selected_venues)}")

    df = df[df['VenueID'].isin(venue2pid)].copy()
    df['Pid'] = df['VenueID'].map(venue2pid)

    # 按时间排序（保持签到时序）
    try:
        df['TS'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
    except Exception:
        df['TS'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.sort_values(by=['UserID', 'TS'], kind='mergesort').reset_index(drop=True)

    # 每个用户一条序列，保留>=3次签到的用户
    seqs = []
    for uid, grp in df.groupby('UserID'):
        seq = grp['Pid'].tolist()
        if len(seq) >= 3:
            seqs.append(seq)
    print(f"[序列构建] 用户数: {len(seqs)}, 总签到数: {sum(len(s) for s in seqs)}")
    return seqs


# ============================================================
#  第二步：RQVAE训练（与已接受的最终配置完全一致）
# ============================================================
def train_rqvae_model(model, loader, device, is_gad, epochs=120, lr=8e-4,
                      weight_decay=1e-4, warmup_epochs=8):
    """与pilot_experiment.train_model一致的训练循环（无评估、保留最佳权重）"""
    from torch.optim import AdamW
    from torch.optim import lr_scheduler

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_steps_per_epoch = len(loader)
    warmup_steps = warmup_epochs * num_steps_per_epoch

    def _lr_lambda(step):
        return float(step) / float(max(1, warmup_steps)) if step < warmup_steps else 1.0

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    for epoch in range(epochs):
        model.train()
        total_loss, n_b = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=False):
            optimizer.zero_grad()
            if len(batch) == 4:
                pids, features, lats, lons = batch
                lats, lons = lats.to(device), lons.to(device)
            else:
                pids, features = batch
                lats = lons = None
            features = features.to(device)

            if is_gad:
                out, rq_loss, indices, extra = model(features, epoch, lats, lons)
                loss, loss_recon = model.compute_loss(out, rq_loss, xs=features, geo_loss=extra['geo_loss'])
            else:
                out, rq_loss, indices = model(features, epoch)
                loss, loss_recon = model.compute_loss(out, rq_loss, xs=features)

            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_b += 1
        if (epoch + 1) % 20 == 0:
            print(f"  [Epoch {epoch + 1}] loss={total_loss / max(n_b, 1):.4f}")
    return model


@torch.no_grad()
def extract_sids(model, loader, device, is_gad):
    """提取所有POI的SID索引矩阵 (N, 3)"""
    model.eval()
    all_indices = []
    for batch in loader:
        if len(batch) == 4:
            _, features, _, _ = batch
        else:
            _, features = batch
        features = features.to(device)
        if is_gad:
            x_q, indices, distances, x_e = model.get_indices(features, 0)
        else:
            x_q, indices, distances = model.get_indices(features, 0)
        all_indices.append(indices.view(-1, indices.shape[-1]).cpu())
    return torch.cat(all_indices, dim=0)  # (N, 3)


# ============================================================
#  第三步：SID序列模型（GRU，作为原论文LLM生成器的轻量代理）
# ============================================================
class SIDSeqDataset(Dataset):
    """历史POI序列（转SID索引）→ 下一个POI"""

    def __init__(self, samples, sid_matrix):
        self.samples = samples  # list of (hist_pids:list, target_pid:int)
        self.sid = sid_matrix  # (num_pois, 3) LongTensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, target = self.samples[idx]
        return self.sid[hist], len(hist), target  # (T,3), T, int


def collate_seq(batch):
    sids, lens, targets = zip(*batch)
    max_len = max(lens)
    B, L = len(batch), sids[0].shape[-1]
    padded = torch.zeros(B, max_len, L, dtype=torch.long)
    for i, (s, ln) in enumerate(zip(sids, lens)):
        padded[i, :ln] = s
    return padded, torch.tensor(lens, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class SIDSeqModel(nn.Module):
    """POI表示 = 3级SID码embedding之和；GRU编码历史 → 输出query → 与全部候选POI表示做点积检索"""

    def __init__(self, num_codes, num_pois, num_levels=3, emb_dim=64, hidden=128):
        super().__init__()
        self.num_levels = num_levels
        self.code_embs = nn.ModuleList([nn.Embedding(num_codes, emb_dim) for _ in range(num_levels)])
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.out_proj = nn.Linear(hidden, emb_dim)

    def poi_reps(self, sid_matrix):
        # sid_matrix: (P, L) → (P, emb_dim)
        reps = 0
        for l in range(self.num_levels):
            reps = reps + self.code_embs[l](sid_matrix[:, l])
        return reps  # (P, D)

    def forward(self, sid_hist, lens, sid_matrix):
        # sid_hist: (B, T, L); lens: (B,); sid_matrix: (P, L)
        B, T, L = sid_hist.shape
        x = 0
        for l in range(L):
            x = x + self.code_embs[l](sid_hist[:, :, l])  # (B,T,D)
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)          # (1, B, hidden)
        q = self.out_proj(h_n.squeeze(0))  # (B, D)
        reps = self.poi_reps(sid_matrix)   # (P, D)
        scores = q @ reps.t()              # (B, P)
        return scores


def train_sid_seq_model(sid_matrix, train_samples, val_samples, num_pois, num_codes, device,
                        epochs=15, lr=1e-3, batch_size=512, hidden=128, seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SIDSeqModel(num_codes=num_codes, num_pois=num_pois, hidden=hidden).to(device)
    train_dataset = SIDSeqDataset(train_samples, sid_matrix)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    sid_matrix = sid_matrix.to(device)

    best_val_acc = 0.0
    best_state = None
    patience, no_improve = 5, 0

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for sid_hist, lens, targets in tqdm(loader, desc=f"SeqEpoch {epoch + 1}/{epochs}", ncols=100, leave=False):
            sid_hist, lens, targets = sid_hist.to(device), lens, targets.to(device)
            scores = model(sid_hist, lens, sid_matrix)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        # 验证集评估（用于早停与模型选择）
        val_acc = evaluate_sid_seq_model(model, val_samples, sid_matrix, device, ks=(1,))['Acc@1']
        print(f"  [SeqEpoch {epoch + 1}] CE loss={total_loss / max(n, 1):.4f}, Val Acc@1={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [早停] 连续{patience}轮无提升，停止训练")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_sid_seq_model(model, test_samples, sid_matrix, device, ks=(1, 5, 10)):
    model.eval()
    sid_matrix = sid_matrix.to(device)
    dataset = SIDSeqDataset(test_samples, sid_matrix)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_seq)
    hits = {k: 0 for k in ks}
    total = 0
    for sid_hist, lens, targets in loader:
        sid_hist, lens, targets = sid_hist.to(device), lens, targets.to(device)
        scores = model(sid_hist, lens, sid_matrix)
        _, topk = torch.topk(scores, max(ks), dim=1)  # (B, max_k)
        for k in ks:
            hits[k] += (topk[:, :k] == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.shape[0]
    return {f"Acc@{k}": hits[k] / max(total, 1) for k in ks}


# ============================================================
#  主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下游Acc@1评估：Baseline SID vs GAD SID")
    parser.add_argument('--num_pois', type=int, default=600)
    parser.add_argument('--dataset', type=str, default='NYC', choices=['NYC', 'TKY'], help='选择TSMC2014数据集')
    parser.add_argument('--geo_loss_weight', type=float, default=0.03, help='geo损失权重（NYC=0.03/TKY=0.02）')
    parser.add_argument('--geo_warmup', type=int, default=100, help='geo损失warmup epoch数')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[128, 128, 128], help='codebook规模（TKY用256 256 256）')
    parser.add_argument('--pe_dim', type=int, default=32, help='正弦PE单坐标维度（2*pe_dim为增量）')
    parser.add_argument('--rqvae_epochs', type=int, default=120)
    parser.add_argument('--seq_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_hist', type=int, default=10, help='序列模型最大历史长度')
    args = parser.parse_args()

    device = torch.device(args.device)
    tsmc_path = os.path.join(ROOT, f'dataset_TSMC2014_{args.dataset}.txt')
    # 目录名含数据集关键词 → GeoEmbDataset自动匹配维度配置（NYC/TKY）
    subset_dir = os.path.join(ROOT, 'datasets', f'pilot_subset_{args.dataset.lower()}')

    # ---------- 1. 数据准备 ----------
    info_path, coords_path = build_subset_data(tsmc_path, subset_dir, num_pois=args.num_pois)
    seqs = build_user_sequences(tsmc_path, num_pois=args.num_pois)

    # 80%训练 / 10%验证 / 10%测试划分（与原论文一致）
    # 从所有用户序列生成全部 (history, next) 样本对，再随机划分
    all_samples = []
    for seq in seqs:
        for i in range(1, len(seq)):
            hist = seq[max(0, i - args.max_hist):i]
            all_samples.append((hist, seq[i]))

    # 固定种子打乱后按比例切分
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_total = len(all_samples)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    print(f"[划分 80/10/10] 总样本: {n_total}, 训练: {len(train_samples)}, 验证: {len(val_samples)}, 测试: {len(test_samples)}")

    # ---------- 2. 训练两个RQVAE（与已接受配置完全一致） ----------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_dataset = GeoEmbDataset(info_path, poi_coords_path=coords_path, pe_dim=args.pe_dim, use_geo_features=False)
    gad_dataset = GeoEmbDataset(info_path, poi_coords_path=coords_path, pe_dim=args.pe_dim, use_geo_features=True)

    base_input_dim = base_dataset.get_input_dim()
    gad_input_dim = gad_dataset.get_input_dim()
    print(f"Baseline输入维度: {base_input_dim}, GAD输入维度: {gad_input_dim}")

    base_loader = DataLoader(base_dataset, batch_size=args.batch_size, shuffle=True)
    gad_loader = DataLoader(gad_dataset, batch_size=args.batch_size, shuffle=True)
    base_loader_eval = DataLoader(base_dataset, batch_size=args.batch_size, shuffle=False)
    gad_loader_eval = DataLoader(gad_dataset, batch_size=args.batch_size, shuffle=False)

    num_emb_list = args.num_emb_list
    common_kwargs = dict(
        num_emb_list=num_emb_list, e_dim=64, dropout_prob=0.1, bn=False, loss_type="mse",
        quant_loss_weight=1.0, kmeans_init=False, kmeans_iters=50,
        sk_epsilons=[0.003, 0.003, 0.009], sk_iters=50, use_sk=True,
        use_linear=0, beta=0.25, diversity_loss=0.25,
    )

    print("\n" + "#" * 70)
    print("  [1/3] 训练 Baseline RQVAE")
    print("#" * 70)
    baseline_model = BaselineRQVAE(in_dim=base_input_dim, layers=[256, 128, 64], **common_kwargs)
    baseline_model = train_rqvae_model(baseline_model, base_loader, device, is_gad=False, epochs=args.rqvae_epochs)

    print("\n" + "#" * 70)
    print("  [2/3] 训练 GAD-RQVAE")
    print("#" * 70)
    gad_model = GADRQVAE(in_dim=gad_input_dim, layers=[384, 256, 128],
                         use_geo_loss=True, geo_loss_weight=args.geo_loss_weight, geo_margin=0.1,
                         num_triplets_per_batch=80, geo_warmup_epochs=args.geo_warmup, **common_kwargs)
    gad_model = train_rqvae_model(gad_model, gad_loader, device, is_gad=True, epochs=args.rqvae_epochs)

    # ---------- 3. 提取SID ----------
    sid_base = extract_sids(baseline_model, base_loader_eval, device, is_gad=False)
    sid_gad = extract_sids(gad_model, gad_loader_eval, device, is_gad=True)
    print(f"\n[SID] Baseline shape={tuple(sid_base.shape)}, GAD shape={tuple(sid_gad.shape)}")

    # ---------- 4. 训练SID序列模型并评估 ----------
    print("\n" + "#" * 70)
    print("  [3/3] 训练 SID 序列模型 (GRU) 并评估下游 Acc@k")
    print("#" * 70)

    print("\n>> Baseline SID 序列模型:")
    model_base = train_sid_seq_model(sid_base, train_samples, val_samples, args.num_pois, num_emb_list[0], device,
                                     epochs=args.seq_epochs, seed=args.seed)
    acc_base = evaluate_sid_seq_model(model_base, test_samples, sid_base, device)

    print("\n>> GAD SID 序列模型:")
    model_gad = train_sid_seq_model(sid_gad, train_samples, val_samples, args.num_pois, num_emb_list[0], device,
                                    epochs=args.seq_epochs, seed=args.seed)
    acc_gad = evaluate_sid_seq_model(model_gad, test_samples, sid_gad, device)

    # Popularity参考基线（训练集中最热门POI）
    pop = np.zeros(args.num_pois)
    for _, t in train_samples:
        pop[t] += 1
    top1_pop = int(np.argmax(pop))
    pop_order = np.argsort(-pop)
    hits = {1: 0, 5: 0, 10: 0}
    for _, t in test_samples:
        for k in (1, 5, 10):
            if t in pop_order[:k]:
                hits[k] += 1
    acc_pop = {f"Acc@{k}": hits[k] / len(test_samples) for k in (1, 5, 10)}

    # ---------- 5. 汇总输出 ----------
    print("\n" + "=" * 70)
    print(f"  下游推荐指标 (POI subset = {args.num_pois}, 序列模型 = GRU代理)")
    print("=" * 70)
    header = f"{'指标':<12} | {'MostPop':>10} | {'Baseline':>10} | {'GAD-RQVAE':>10} | {'GAD vs Baseline':>16}"
    print(header)
    print("-" * len(header))
    rows = []
    for k in (1, 5, 10):
        key = f"Acc@{k}"
        delta = (acc_gad[key] - acc_base[key]) / max(acc_base[key], 1e-8) * 100
        print(f"{key:<12} | {acc_pop[key]:>10.4f} | {acc_base[key]:>10.4f} | {acc_gad[key]:>10.4f} | {delta:>+15.2f}%")
        rows.append([key, acc_pop[key], acc_base[key], acc_gad[key], delta])

    print("\n  结论：GAD-RQVAE 的地理感知SID在下游next-POI推荐" +
          ("达到/超过" if acc_gad["Acc@1"] >= acc_base["Acc@1"] else "未达到") +
          " 原方法的 Acc@1 水平。")

    out_csv = os.path.join(subset_dir, 'acc_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'MostPop', 'Baseline', 'GAD-RQVAE', 'GAD_vs_Baseline_%'])
        w.writerows(rows)
    print(f"[完成] 结果已保存至: {out_csv}")


if __name__ == '__main__':
    main()
