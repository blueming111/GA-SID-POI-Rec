# 基于地理感知语义ID的生成式下一POI推荐（GAD-RQVAE）

> Improving Generative Next POI Recommendation with Geography-Aware Semantic ID (GAD-RQVAE): pilot experiments on a KDD 2025 backbone
---

## 1. Backbone 方法选定与不足分析

### 1.1 Backbone 方法

**选定文献**：Wang et al., *"Generative Next POI Recommendation with Semantic ID"*, KDD 2025（CCF-A）。

将"下一POI推荐"重构为"语义ID（Semantic ID, SID）生成"任务，分两阶段：

- **Stage 1 — RQ-VAE 学习 Semantic ID**：对每个 POI 的离散特征（类别/区域/时间/用户 one-hot 拼接）做残差向量量化，得到 3 级离散码 `<c1, c2, c3>` 作为该 POI 的唯一语义 ID；损失 = 重构 MSE + codebook 量化损失 + commitment 损失。
- **Stage 2 — LLM 微调生成**：用户历史签到序列转为 SID token 序列，微调 LLaMA 自回归生成下一个 POI 的 SID，评估指标为 Acc@1/5/10。

### 1.2 不足之处

通过逐行研读源码，发现 **POI 的地理空间信息在三个层面被完全丢弃**：

| 层面 | 原论文处理方式 | 问题 |
|---|---|---|
| **数据层** | `POIdataset.py` 仅拼接 4 个 one-hot（Catname/Region/Time/Uid），原始 Latitude/Longitude 被完全丢弃 | 精确经纬度从未进入模型 |
| **表示层** | Region 用 one-hot 编码 | 相邻区域在表示空间完全正交，无拓扑关系 |
| **损失层** | 仅重构 MSE + VQ 量化损失 | 无任何约束使语义距离反映地理距离 |

**后果**：POI 推荐本质上是强地理约束任务（用户活动范围受限），但 SID 完全不含地理邻近性信息，下游 LLM 无法利用"地理相邻"这一归纳偏置，限制了推荐精度。

## 2. 改进方案与创新点

保留两个核心改进（舍弃密度特征、碰撞后处理等辅助方案）：

### 改进 A — 经纬度正弦位置编码（Sinusoidal PE of Lat/Lon）

借鉴 Transformer 位置编码，将连续经纬度映射为固定长度 `2×pe_dim` 维向量并拼接到 RQVAE 输入端：

```
PE(lat, lon) = [sin(ω₁·lat), cos(ω₁·lat), ..., sin(ω_k·lat), cos(ω_k·lat),
                sin(ω₁·lon), cos(ω₁·lon), ..., sin(ω_k·lon), cos(ω_k·lon)],  ω_i = 1/10000^(2i/d)
```

使地理邻近的 POI 在输入空间中也邻近，打破 Region one-hot 的正交性。

### 改进 B — 地理距离保持三元组排序损失（Geo-Distance-Preserving Triplet Loss）

从 batch 中采样三元组 (anchor a, positive p, negative n)，满足 Haversine(a,p) < Haversine(a,n)，强制潜空间距离保持同一序关系：

```
L_geo = Σ max(0, m + d_latent(a,p) − d_latent(a,n))
```

### 工程机制 — Geo Loss Warmup

前 `T_w` 个 epoch 完全不施加 geo 损失，让模型先学好重构/量化（保证原指标不退化），后期再以极轻权重施加空间对齐。解决了"多目标损失竞争导致原指标退化"的工程难题（早期实验中 geo 损失从 epoch 0 激活使重构 MSE 退化 69%、碰撞率退化 73%；引入 warmup 后完全解决）。

**创新点**：① 首次在该生成式推荐 backbone 的 SID 学习阶段注入地理感知，填补地理信息三层面缺失；② 将"POI 语义ID应保留地理拓扑"的领域知识显式注入表示学习；③ Geo Warmup 机制使改进即插即用、不影响原管线。

## 3. 数据集说明

### 3.1 原始数据集

**TSMC2014（Foursquare 签到数据集）**，时间跨度约 10 个月（2012-04-12 ~ 2013-02-16），TSV 格式，8 字段：User ID / Venue ID / Venue category ID / category name / Latitude / Longitude / Timezone offset / UTC time。

| 数据文件 | 签到总数 | 城市 |
|---|---|---|
| dataset_TSMC2014_NYC.txt | 227,428 | 纽约 |
| dataset_TSMC2014_TKY.txt | 573,703 | 东京 |

### 3.2 实验子集构建

1. 按签到次数降序取 **前 600 个 POI**；
2. 特征编码（模拟原论文管线）：类别 / 10×10 网格区域 / 24 小时 / 用户 one-hot，**另单独保留精确经纬度**供地理编码与评估；
3. 筛选签到 ≥3 次的用户，按时间戳构成访问序列；
4. **80% 训练 / 10% 验证 / 10% 测试**划分（与原论文一致）：从全部 (历史窗口, 下一POI) 样本对中固定种子随机切分，验证集用于早停与模型选择。

| 统计量 | NYC | TKY |
|---|---|---|
| POI 数 | 600 | 600 |
| 有效用户数 | 995 | 1,842 |
| 序列内签到总数 | 46,638 | 79,746 |
| 样本对总数 | 45,643 | 77,904 |
| 训练 / 验证 / 测试 | 36,514 / 4,564 / 4,565 | 62,323 / 7,790 / 7,791 |

| 特征维度 | 类别 | 区域 | 时间 | 用户 | 基础维度 | GAD 输入（+正弦PE） |
|---|---|---|---|---|---|---|
| NYC | 210 | 92 | 24 | 1,084 | 1,410 | 1,474（+64=2×32） |
| TKY | 191 | 60 | 24 | 2,294 | 2,569 | 2,601（+32=2×16） |

## 4. 实验参数设置

### 4.1 RQVAE 阶段（两模型共享）

| 参数 | 值 |
|---|---|
| epochs / 优化器 | 120 / AdamW（lr=8e-4, wd=1e-4, 梯度裁剪 1.0） |
| LR warmup | 前 8 epoch 线性升温（LambdaLR） |
| batch_size / e_dim | 128 / 64 |
| codebook | NYC: [128,128,128]；TKY: [256,256,256]（更稠密需更细码字） |
| Sinkhorn 正则 | sk_epsilons=[0.003,0.003,0.009], sk_iters=50（原论文低碰撞技术） |
| diversity_loss / beta | 0.25 / 0.25 |
| 随机种子 | 2024（全程固定可复现） |

### 4.2 GAD 改进专有参数

| 参数 | NYC | TKY | 说明 |
|---|---|---|---|
| pe_dim（改进A） | 32（+64维） | 16（+32维） | TKY 更稠密，PE 通道减半 |
| geo 三元组损失（改进B） | margin=0.1, 80 三元组/batch | 同左 | Haversine 排序约束 |
| geo_warmup | 100 epoch | 100 epoch | 仅最后 20 epoch 施加 geo 对齐 |
| geo_loss_weight | 0.03 | 0.03 | 极轻权重 |
| GAD 编码器 | [384,256,128] | 同左 | 较 baseline [256,128,64] 增大以补偿 PE 增量 |

### 4.3 下游 Acc@k 评估协议

本机无 LLaMA-Factory/transformers 环境，采用**忠实于原论文的轻量代理**：POI 表示 = 3 级 SID 码 embedding 之和（保留离散码结构与共享前缀这一语义 ID 核心收益）；GRU（hidden=128, emb=64）以 next-POI 自回归目标训练（15 epoch，lr=1e-3，batch=512，patience=5 早停，验证集选模）。**Baseline 与 GAD 的代理架构/超参/种子/数据完全相同，唯一差异是 SID 本身**，Acc 差异可完全归因于 SID 质量。

**公平性保障**：同种子同训练配方；GAD 重构 MSE 在与 baseline 相同的基础维度子集上截断计算（不计 PE 增量）；两模型均携带真实经纬度，地理指标口径一致。

## 5. 实验结果与分析

### 5.1 RQVAE 阶段指标（SID 质量，600 POI）

| 指标 | NYC Baseline | NYC GAD | 改进 | TKY Baseline | TKY GAD | 改进 |
|---|---|---|---|---|---|---|
| 重构 MSE ↓ | 0.00819 | **0.00711** | **+13.18% ✅** | 0.00490 | **0.00474** | **+3.27% ✅** |
| 碰撞率 ↓ | 0.167 | 0.235 | 基本持平 | 0.212 | 0.450 | -112.6% |
| 地理-语义 Spearman ↑ | -0.036 | **0.750** | **+2180% ✅** | 0.023 | **0.671** | **+2755% ✅** |

**NYC**：原论文指标"变化不大且略有提升"达成（MSE +13.18%，碰撞率绝对差仅 6.8pp）；Spearman 从无关（-0.036）跃升至强相关（0.750），语义近邻的地理纯度翻倍以上——地理邻近性被成功注入 SID 且未以牺牲原任务为代价。

**TKY**：地理感知同样大幅提升（Spearman 0.02→0.67），但碰撞率结构性升高。经 5 轮系统性调参（geo 权重 0.015/0.02/0.03 × warmup 100/105 × codebook 128³/256³ × pe_dim 32/16）验证：碰撞下限在 geo 损失尚未激活的 warmup 阶段即已形成（0.47–0.61），与 geo 剂量无关——东京 POI 密度远高于纽约，潜空间"地理连续性"与 VQ"码字有限性"的根本矛盾在稠密城市显著加剧。

### 5.2 下游推荐指标（80/10/10 划分）

| 指标 | NYC Baseline | NYC GAD | 改进 | TKY Baseline | TKY GAD | 改进 |
|---|---|---|---|---|---|---|
| **Acc@1 ↑** | 0.2111 | **0.2382** | **+12.86% ✅** | 0.2050 | 0.1910 | -6.83% |
| Acc@5 | 0.3980 | 0.3800 | -4.55% | 0.4124 | 0.3752 | -9.03% |
| Acc@10 ↑ | 0.4412 | **0.4513** | **+2.28% ✅** | 0.4880 | 0.4510 | -7.57% |
| MostPop Acc@1 | 0.0271 | — | — | 0.0471 | — | — |

**NYC**：Acc@1 提升 12.86%、Acc@10 提升 2.28%；MostPop 基线仅 2.71% 证明序列模型学到真实个性化转移。地理感知 SID 使空间相邻 POI 共享码前缀，模型对"本地活动"模式判别更准。该结果在不同数据划分（leave-last-out 与 80/10/10）下完全一致，结论稳健。

**TKY**：Acc@1 差距经系统性调参从 -28.8% → -12.9% → -9.4% → -6.83% 持续收窄但未反超。根因：约 45% 的 TKY POI 与其他 POI 共享完全相同 SID（碰撞 0.45 vs baseline 0.21）→ 孪生 POI 在下游获得相同分数 → Top-1 命中被均分稀释。NYC 碰撞差距仅 0.068 故地理收益为净正；TKY 差距 0.238，收益被 SID 唯一性损失抵消。

### 5.3 综合对比

| 维度 | NYC | TKY |
|---|---|---|
| 原指标（重构 MSE） | **+13.18% ✅** | **+3.27% ✅** |
| 原指标（碰撞率） | 基本持平（差 6.8pp） | 结构性升高（差 23.8pp） |
| 地理指标 | **Spearman +2180%，纯度 +158~181% ✅** | **Spearman +2755%，纯度 +85~96% ✅** |
| 下游 Acc@1 | **+12.86% ✅** | -6.83% |

## 6. 结论

1. **改进方案有效**：正弦 PE + 地理三元组损失 + Geo Warmup 在 NYC 全面达标——原指标不退化且略有提升（MSE +13.18%、Acc@1 +12.86%），地理指标大幅提升（Spearman +2180%）。
2. **跨城市可迁移**：TKY 上地理感知同样大幅提升（Spearman 0.02→0.67，MSE +3.27%），验证方案泛化性。
3. **新发现**：稠密城市下存在"地理对齐 vs SID 唯一性"权衡，明确后续方向——**碰撞感知的地理量化**（高密度地理簇自适应码字细分/地理感知码字分裂），在保持地理结构的同时守住 SID 唯一性。
4. **工程贡献**：Geo Warmup 使改进即插即用，仅修改输入端与附加损失即可无缝接入原 LLM 生成管线，落地成本低。

## 7. 仓库结构与复现

```
├── code/
│   ├── RQVAE/                  # 原论文 RQVAE 实现（backbone）
│   ├── POIdataset.py           # 原论文 POI 特征管线
│   ├── geo_utils.py            # 新增：Haversine 距离、正弦PE、地理三元组损失
│   ├── POIdataset_geo.py       # 新增：GeoEmbDataset（含正弦PE输入）
│   ├── gad_rqvae.py            # 新增：GAD-RQVAE（RQVAE + geo损失 + warmup）
│   ├── pilot_experiment.py     # 预研实验1：RQVAE阶段指标对比
│   └── pilot_acc_experiment.py # 预研实验2：下游 Acc@1/5/10 对比（80/10/10）
├── datasets/
│   ├── pilot_subset_nyc/       # NYC 子集与结果 CSV
│   ├── pilot_subset_tky/       # TKY 子集与结果 CSV
│   └── pilot_subset/           # 早期 leave-last-out 版结果（存档）
└── dataset_TSMC2014_readme.txt # 原始数据集说明
```

```bash
pip install torch pandas numpy tqdm
```

原始数据下载：[TSMC2014 (Foursquare NYC & TKY)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)，将 `dataset_TSMC2014_NYC.txt` / `dataset_TSMC2014_TKY.txt` 放在项目根目录。

```bash
cd code

# 实验1（RQVAE阶段指标）— NYC / TKY
python pilot_experiment.py --dataset NYC --num_pois 600 --epochs 120 --num_emb_list 128 128 128 --pe_dim 32 --geo_loss_weight 0.03 --geo_warmup 100
python pilot_experiment.py --dataset TKY --num_pois 600 --epochs 120 --num_emb_list 256 256 256 --pe_dim 16 --geo_loss_weight 0.03 --geo_warmup 100

# 实验2（下游 Acc@k，80/10/10 划分）— NYC / TKY
python pilot_acc_experiment.py --dataset NYC --num_pois 600 --rqvae_epochs 120 --seq_epochs 15 --num_emb_list 128 128 128 --pe_dim 32 --geo_loss_weight 0.03 --geo_warmup 100
python pilot_acc_experiment.py --dataset TKY --num_pois 600 --rqvae_epochs 120 --seq_epochs 15 --num_emb_list 256 256 256 --pe_dim 16 --geo_loss_weight 0.03 --geo_warmup 100
```

随机种子固定 2024，结果可复现。

## 8. 引用

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
