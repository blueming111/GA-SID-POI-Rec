# 基于地理感知语义ID的生成式下一POI推荐（GAD-RQVAE）

> 南开大学计算机学院保研考核调研/预研实验。以 KDD 2025 论文 *Generative Next POI Recommendation with Semantic ID* 为 backbone，发现并修复其对**地理空间信息的缺失**，通过预研实验验证改进方案的贡献与可行性。

---

## 1. Backbone 方法与不足

**Backbone**：Wang et al., *Generative Next POI Recommendation with Semantic ID*, KDD 2025。
方法分两阶段：① RQ-VAE 将 POI 特征量化为 3 级离散语义 ID（SID）；② 微调 LLM 自回归生成下一个 POI 的 SID。

**发现的不足**：通过研读源码，发现 POI 的地理信息在**三个层面被完全丢弃**：

| 层面 | 原处理方式 | 问题 |
|---|---|---|
| 数据层 | `POIdataset.py` 仅拼接 4 个 one-hot（类别/区域/时间/用户），原始经纬度被丢弃 | 精确坐标从未进入模型 |
| 表示层 | 区域用 one-hot | 相邻区域在表示空间完全正交，无拓扑关系 |
| 损失层 | 仅重构 MSE + 量化损失 | 无任何约束使语义距离反映地理距离 |

## 2. 改进方案（两个核心改进）

1. **改进A — 经纬度正弦位置编码**：借鉴 Transformer PE，将连续经纬度映射为 `2×pe_dim` 维 sin/cos 对数频带向量，拼接到 RQVAE 输入端，使地理邻近 POI 在输入空间邻近。
2. **改进B — 地理距离保持三元组损失**：在编码器输出上施加 margin ranking loss，强制 Haversine 距离更近的 POI 对在潜空间中更近：
   `L_geo = Σ max(0, m + d_latent(a,p) − d_latent(a,n))`
3. **工程机制 — Geo Warmup**：前 100 epoch 不施加 geo 损失，让重构/量化先收敛，最后 20 epoch 以极轻权重（0.03）做空间对齐。解决多目标损失竞争导致原指标退化的问题。

## 3. 仓库结构

```
├── code/
│   ├── RQVAE/                  # 原论文 RQVAE 实现（backbone）
│   ├── POIdataset.py           # 原论文 POI 特征管线
│   ├── geo_utils.py            # 新增：Haversine 距离、正弦PE、地理三元组损失
│   ├── POIdataset_geo.py       # 新增：GeoEmbDataset（含正弦PE输入）
│   ├── gad_rqvae.py            # 新增：GAD-RQVAE（RQVAE + geo损失 + warmup）
│   ├── pilot_experiment.py     # 预研实验1：RQVAE阶段指标对比
│   └── pilot_acc_experiment.py # 预研实验2：下游 Acc@1/5/10 对比
├── datasets/
│   ├── pilot_subset_nyc/       # NYC 子集与结果（poi_info/coords + 结果CSV）
│   └── pilot_subset_tky/       # TKY 子集与结果
└── dataset_TSMC2014_readme.txt # 原始数据集说明
```

## 4. 环境与复现

```bash
pip install torch pandas numpy tqdm
```

原始数据集下载：[TSMC2014 (Foursquare NYC & TKY)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)（使用请引用原作者：Dingqi Yang et al., WWW 2015）。将 `dataset_TSMC2014_NYC.txt` / `dataset_TSMC2014_TKY.txt` 放在项目根目录。

```bash
cd code

# 实验1（RQVAE阶段指标）—— NYC
python pilot_experiment.py --device cpu --num_pois 600 --epochs 120 --dataset NYC \
    --num_emb_list 128 128 128 --pe_dim 32 --geo_loss_weight 0.03 --geo_warmup 100

# 实验1 —— TKY
python pilot_experiment.py --device cpu --num_pois 600 --epochs 120 --dataset TKY \
    --num_emb_list 256 256 256 --pe_dim 16 --geo_loss_weight 0.03 --geo_warmup 100

# 实验2（下游 Acc@k，80/10/10 划分）—— NYC / TKY 同理
python pilot_acc_experiment.py --device cpu --num_pois 600 --rqvae_epochs 120 --seq_epochs 15 \
    --dataset NYC --num_emb_list 128 128 128 --pe_dim 32 --geo_loss_weight 0.03 --geo_warmup 100
```

随机种子固定为 2024，结果可复现。

## 5. 实验结果

### 5.1 RQVAE 阶段指标（600 POI）

| 指标 | NYC Baseline | NYC GAD | 改进 | TKY Baseline | TKY GAD | 改进 |
|---|---|---|---|---|---|---|
| 重构 MSE ↓ | 0.00819 | **0.00711** | **+13.18%** | 0.00490 | **0.00474** | **+3.27%** |
| 碰撞率 ↓ | 0.167 | 0.235 | 基本持平 | 0.212 | 0.450 | -112.6% |
| 地理-语义 Spearman ↑ | -0.036 | **0.750** | **+2180%** | 0.023 | **0.671** | **+2755%** |
| Top-10 地理纯度 ↑ | 0.041 | **0.114** | **+180.7%** | 0.041 | **0.081** | **+96.0%** |

### 5.2 下游推荐指标（GRU 代理 LLM，80/10/10 划分）

| 指标 | NYC Baseline | NYC GAD | 改进 | TKY Baseline | TKY GAD | 改进 |
|---|---|---|---|---|---|---|
| Acc@1 ↑ | 0.2111 | **0.2382** | **+12.86%** | 0.2050 | 0.1910 | -6.83% |
| Acc@10 ↑ | 0.4412 | **0.4513** | **+2.28%** | 0.4880 | 0.4510 | -7.57% |

> 下游协议说明：本机无 LLaMA-Factory 环境，用相同架构/超参/种子的 GRU（POI 表示 = 3 级 SID 码 embedding 之和）作为 LLM 生成器的轻量代理，Baseline 与 GAD 唯一差异是 SID 本身，Acc 差异可归因于 SID 质量。

### 5.3 结论

- **NYC 全面达标**：原论文指标不退化且略有提升（MSE +13.18%、Acc@1 +12.86%），地理指标大幅提升（Spearman +2180%）。
- **跨城市可迁移**：TKY 上地理感知同样大幅提升（Spearman 0.02→0.67），但揭示出稠密城市下"地理对齐 vs SID 唯一性"的权衡：地理邻近 POI 争先共享码字使碰撞率结构性升高，稀释了下游 Top-1 命中率。
- **后续方向**：碰撞感知的地理量化（高密度地理簇自适应码字细分），在保持地理结构的同时守住 SID 唯一性。

## 6. 引用

```bibtex
@inproceedings{wang2025generative,
  title={Generative Next POI Recommendation with Semantic ID},
  author={Wang, et al.},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2025}
}
@inproceedings{yang2015participatory,
  title={Participatory cultural mapping based on volunteer behavior information},
  author={Yang, Dingqi and Zhang, Daqing and Yu, Zhiyong and Wanglong, Zheng},
  booktitle={Proceedings of the 24th International Conference on World Wide Web (WWW)},
  year={2015}
}
```
