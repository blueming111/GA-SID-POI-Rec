"""
GAD-RQVAE: Geography-Aware Disentangled Residual Vector Quantized VAE

在原版RQVAE基础上新增地理距离保持损失（Geo-Preserving Loss）
"""
import torch
from torch import nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RQVAE.layers import MLPLayers
from RQVAE.rq import ResidualVectorQuantizer
from geo_utils import geo_distance_preserving_loss


class GADRQVAE(nn.Module):
    def __init__(self,
                 in_dim=16,
                 num_emb_list=[32, 32, 32],
                 e_dim=64,
                 layers=[128, 64],
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=20,
                 sk_epsilons=None,
                 sk_iters=20,
                 use_sk=False,
                 use_linear=0,
                 beta=0.25,
                 diversity_loss=0.0,
                 # ---- GAD新增参数 ----
                 use_geo_loss=True,
                 geo_loss_weight=1.0,
                 geo_margin=0.1,
                 num_triplets_per_batch=64,
                 geo_warmup_epochs=30,  # 新增：Geo损失warmup epoch数
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight

        # 地理损失参数
        self.use_geo_loss = use_geo_loss
        self.geo_loss_weight = geo_loss_weight
        self.geo_margin = geo_margin
        self.num_triplets_per_batch = num_triplets_per_batch
        self.geo_warmup_epochs = geo_warmup_epochs

        # 编码器
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,
                                 bn=self.bn,
                                 weight_init='xavier')

        # 残差VQ
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons if sk_epsilons is not None else [0.0] * len(num_emb_list),
            sk_iters=sk_iters,
            use_sk=use_sk,
            use_linear=use_linear,
            diversity_loss=diversity_loss
        )

        # 解码器
        self.decode_layer_dims = [self.e_dim] + self.layers + [self.in_dim]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
            weight_init='xavier')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, epoch_idx, lats=None, lons=None):
        """
        :param x: (B, in_dim) 输入特征
        :param epoch_idx: 当前epoch（用于diversity_loss的warmup）
        :param lats: (B,) 纬度（仅训练时需要，用于geo loss）
        :param lons: (B,) 经度（仅训练时需要，用于geo loss）
        :return: out, rq_loss, indices, extra_loss_dict
        """
        x_e = self.encoder(x)  # (B, e_dim)
        x_q, rq_loss, indices, distances = self.rq(x_e, epoch_idx)
        out = self.decoder(x_q)
        out = self.sigmoid(out)

        # 计算地理距离保持损失（仅当提供经纬度时）
        geo_loss_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.use_geo_loss and lats is not None and lons is not None and self.training:
            # Geo Loss Warmup：前 geo_warmup_epochs 个epoch不施加geo损失，
            # 让模型先学好重构/量化，后面再对齐空间关系
            geo_warmup_epochs = getattr(self, 'geo_warmup_epochs', 30)
            if epoch_idx >= geo_warmup_epochs:
                geo_loss_val = geo_distance_preserving_loss(
                    x_e, lats, lons,
                    margin=self.geo_margin,
                    num_triplets=self.num_triplets_per_batch
                )

        extra = {"geo_loss": geo_loss_val}
        return out, rq_loss, indices, extra

    @torch.no_grad()
    def get_indices(self, xs, epoch_idx=0):
        x_e = self.encoder(xs)
        x_q, _, indices, distances = self.rq(x_e, epoch_idx)
        return x_q, indices, distances, x_e  # 返回x_e用于评估空间相关性

    def compute_loss(self, out, quant_loss, xs=None, geo_loss=None):
        if self.loss_type == 'mse':
            loss_recon = nn.functional.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = nn.functional.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        if self.use_geo_loss and geo_loss is not None:
            loss_total = loss_total + self.geo_loss_weight * geo_loss

        return loss_total, loss_recon
