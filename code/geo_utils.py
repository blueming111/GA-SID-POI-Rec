"""
地理计算工具函数：
1. Haversine球面距离计算
2. 经纬度的正弦/余弦位置编码（类似Transformer PE）
3. 局部地理密度特征计算
"""
import numpy as np
import torch


def haversine_distance(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """
    计算两点间的Haversine球面距离（单位：km）
    :param lat1, lon1: 点1的纬度、经度（十进制度）
    :param lat2, lon2: 点2的纬度、经度（十进制度）
    :return: 距离（km）
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c


def haversine_distance_batch(lats, lons):
    """
    批量计算两两POI间的Haversine距离矩阵
    :param lats: (N,) 纬度数组
    :param lons: (N,) 经度数组
    :return: (N, N) 距离矩阵（km）
    """
    N = len(lats)
    lats_rad = np.radians(lats.values if hasattr(lats, 'values') else lats)
    lons_rad = np.radians(lons.values if hasattr(lons, 'values') else lons)

    dlat = lats_rad[:, None] - lats_rad[None, :]
    dlon = lons_rad[:, None] - lons_rad[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats_rad[:, None]) * np.cos(lats_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c


def sinusoidal_encoding(lat, lon, pe_dim=16, max_scale=180.0):
    """
    对经纬度进行正弦/余弦位置编码
    :param lat: 纬度（标量或数组），范围 [-90, 90]
    :param lon: 经度（标量或数组），范围 [-180, 180]
    :param pe_dim: 每个坐标的编码维度（总输出维度=2*pe_dim）
    :param max_scale: 最大坐标绝对值（用于归一化）
    :return: 编码后的向量，shape=(..., 2*pe_dim)
    """
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)

    # 归一化到 [-1, 1]
    lat_norm = lat / 90.0
    lon_norm = lon / max_scale

    # 计算频率
    div_term = np.exp(np.arange(0, pe_dim, 2, dtype=np.float32) * (-np.log(10000.0) / pe_dim))

    # 分别对lat和lon做正弦余弦编码
    def _encode(x_norm):
        # x_norm shape: (...,)
        # 扩展维度以便广播
        x_expanded = x_norm[..., None]  # (..., 1)
        pe_sin = np.sin(x_expanded * div_term * max_scale * np.pi)  # (..., pe_dim//2)
        pe_cos = np.cos(x_expanded * div_term * max_scale * np.pi)
        # 交错拼接，最终形状 (..., pe_dim)
        pe = np.concatenate([pe_sin, pe_cos], axis=-1)
        return pe

    pe_lat = _encode(lat_norm)
    pe_lon = _encode(lon_norm)
    return np.concatenate([pe_lat, pe_lon], axis=-1).astype(np.float32)


def compute_local_density(lats, lons, radius_km=1.0):
    """
    计算每个POI周围指定半径内的POI数量（局部地理密度）
    :param lats: 纬度数组
    :param lons: 经度数组
    :param radius_km: 半径（km）
    :return: 密度数组，shape同lats
    """
    dist_mat = haversine_distance_batch(lats, lons)
    density = (dist_mat <= radius_km).sum(axis=1) - 1  # 减去自身
    return density.astype(np.float32)


# ---------- PyTorch版损失函数（用于训练GAD-RQVAE）----------
def haversine_distance_torch(lats1, lons1, lats2, lons2, radius_km=6371.0):
    """PyTorch版Haversine距离，支持梯度"""
    lats1 = torch.deg2rad(lats1)
    lons1 = torch.deg2rad(lons1)
    lats2 = torch.deg2rad(lats2)
    lons2 = torch.deg2rad(lons2)
    dlat = lats2 - lats1
    dlon = lons2 - lons1
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lats1) * torch.cos(lats2) * torch.sin(dlon / 2.0) ** 2
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
    return radius_km * c


def geo_distance_preserving_loss(latent_vectors, lats, lons, margin=0.1, num_triplets=None):
    """
    地理距离保持对比损失（Margin Ranking Loss on Triplets）
    
    对batch内采样的三元组 (a, p, n)：
      - 若 d_geo(a, p) < d_geo(a, n) 则要求 d_latent(a, p) < d_latent(a, n) - margin
    
    :param latent_vectors: (B, D) batch内POI的潜在向量
    :param lats: (B,) 纬度
    :param lons: (B,) 经度
    :param margin: 对比间隔
    :param num_triplets: 采样三元组数量，默认 = B*(B-1)*(B-2)/6 的合理缩减
    :return: 标量损失
    """
    B = latent_vectors.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=latent_vectors.device, dtype=latent_vectors.dtype)

    # 计算地理距离矩阵
    lats_e = lats.unsqueeze(0)
    lons_e = lons.unsqueeze(0)
    dlat = torch.deg2rad(lats_e.T - lats_e)
    dlon = torch.deg2rad(lons_e.T - lons_e)
    cos_lat = torch.cos(torch.deg2rad(lats_e))
    a = torch.sin(dlat / 2.0) ** 2 + cos_lat.T * cos_lat * torch.sin(dlon / 2.0) ** 2
    geo_dist = 6371.0 * 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))  # (B, B) km

    # 计算潜在空间距离矩阵
    latent_norm = (latent_vectors ** 2).sum(dim=1, keepdim=True)
    latent_dist = torch.sqrt(
        torch.clamp(latent_norm + latent_norm.T - 2 * latent_vectors @ latent_vectors.T, min=1e-8)
    )  # (B, B)

    # 采样三元组：对每个a随机选p和n使得d_geo(a,p) < d_geo(a,n)
    device = latent_vectors.device
    if num_triplets is None:
        num_triplets = max(B * 5, 32)  # 每个anchor采样约5个三元组

    losses = []
    for _ in range(num_triplets):
        a_idx = torch.randint(0, B, (1,), device=device).item()
        # 采样两个不同的索引
        others = [i for i in range(B) if i != a_idx]
        if len(others) < 2:
            continue
        i, j = np.random.choice(others, 2, replace=False)
        d_geo_ap = geo_dist[a_idx, i]
        d_geo_an = geo_dist[a_idx, j]
        d_lat_ap = latent_dist[a_idx, i]
        d_lat_an = latent_dist[a_idx, j]
        # 保证 p是地理更近者，n是更远者
        if d_geo_ap > d_geo_an:
            i, j = j, i
            d_geo_ap, d_geo_an = d_geo_an, d_geo_ap
            d_lat_ap, d_lat_an = d_lat_an, d_lat_ap
        # Ranking loss: max(0, margin + d_lat_ap - d_lat_an)
        losses.append(torch.clamp(margin + d_lat_ap - d_lat_an, min=0.0))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, dtype=latent_vectors.dtype)
    return torch.stack(losses).mean()
