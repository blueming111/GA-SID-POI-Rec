"""
改进版POI数据集：支持多尺度地理特征融合
在原始EmbDataset基础上新增：
1. 经纬度的正弦余弦位置编码（精确地理空间信息）
2. 局部地理密度特征
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geo_utils import sinusoidal_encoding, compute_local_density


class GeoEmbDataset(Dataset):
    """
    地理增强版POI特征数据集（精简版：仅保留经纬度正弦编码）

    输入特征由以下部分拼接：
    1. 类别one-hot (cat_num)
    2. 区域one-hot (region_num)
    3. 时间one-hot (24)
    4. 访问用户one-hot (neighbor_num)
    5. 经纬度正弦编码 (2*pe_dim)          -- 核心改进1（保留）
    """

    def __init__(self, datapath, poi_coords_path=None, pe_dim=16, use_geo_features=True):
        """
        :param datapath: poi_info.csv路径
        :param poi_coords_path: 包含POI经纬度坐标的csv路径
        :param pe_dim: 单坐标正弦编码维度
        :param use_geo_features: 是否启用地理增强特征（False时退化为原版EmbDataset）
        """
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
        # 兼容绝对路径/相对路径
        if os.path.exists(datapath):
            full_path = datapath
        elif os.path.exists(current_dir + datapath):
            full_path = current_dir + datapath
        else:
            full_path = os.path.join(current_dir, datapath.lstrip('/\\'))
        print(f"[GeoEmbDataset] 加载数据: {full_path}")
        data = pd.read_csv(full_path)
        self.ids = data['Pid']
        data['Uid'] = data['Uid'].apply(eval)
        data['Time'] = data['Time'].apply(eval)
        data['neighbors'] = data['neighbors'].apply(eval)
        data['forward_neighbors'] = data['forward_neighbors'].apply(eval)

        # 灵活解析数据模式：优先从文件名关键词识别
        path_lower = datapath.lower()
        if 'tky' in path_lower:
            mode = 'TKY'
        elif 'nyc' in path_lower:
            mode = 'NYC'
        elif 'ca' in path_lower:
            mode = 'CA'
        else:
            # 使用子集的默认配置
            mode = 'NYC'
            print(f"[GeoEmbDataset] 路径无模式关键词，默认采用 NYC 维度配置")
        time_num = 24
        if mode == 'NYC':
            cat_num = 210
            region_num = 92
            neighbor_num = 1084
        elif mode == 'TKY':
            cat_num = 191
            region_num = 60
            neighbor_num = 2294
        elif mode == 'CA':
            cat_num = 304
            region_num = 958
            neighbor_num = 6593
        else:
            raise ValueError("Invalid data mode. Choose from 'NYC', 'TKY', or 'CA'.")

        self.mode = mode
        self.cat_num = cat_num
        self.region_num = region_num
        self.time_num = time_num
        self.neighbor_num = neighbor_num
        self.use_geo_features = use_geo_features
        self.pe_dim = pe_dim

        # ----------- 原版特征构建 -----------
        def to_one_hot_fixed_dim(indices, num_classes, scale_factor=1):
            one_hot = torch.zeros(num_classes, dtype=torch.float32)
            # 统一将索引转换为int，避免eval后字符串的问题
            valid_indices = []
            for i in indices:
                try:
                    iv = int(i)
                    if 0 <= iv < num_classes:
                        valid_indices.append(iv)
                except:
                    pass
            if valid_indices:
                one_hot[valid_indices] = 1
            one_hot *= scale_factor
            return one_hot

        categories = []
        for cat in data['Catname']:
            cat = to_one_hot_fixed_dim(cat, cat_num, scale_factor=1)
            categories.append(cat)
        self.categories = categories

        regions = []
        for region in data['Region']:
            region = to_one_hot_fixed_dim(region, region_num, scale_factor=1)
            regions.append(region)
        self.regions = regions

        times = []
        for t in data['Time']:
            t = to_one_hot_fixed_dim(t, time_num, scale_factor=1)
            times.append(t)
        self.times = times

        neighbors = []
        for nb in data['Uid']:
            nb = to_one_hot_fixed_dim(nb, neighbor_num, scale_factor=1)
            neighbors.append(nb)
        self.neighbors = neighbors

        # ----------- 新增：地理特征构建（仅经纬度正弦编码，精简版）-----------
        self.lats = None
        self.lons = None
        self.geo_encodings = None

        if self.use_geo_features:
            coords = None
            if poi_coords_path is not None and os.path.exists(poi_coords_path):
                coords = pd.read_csv(poi_coords_path)
                coords['Pid'] = coords['Pid'].astype(str)
                id_to_lat = dict(zip(coords['Pid'].astype(str), coords['Latitude']))
                id_to_lon = dict(zip(coords['Pid'].astype(str), coords['Longitude']))
                self.lats = np.array([id_to_lat.get(str(pid), 0.0) for pid in self.ids], dtype=np.float32)
                self.lons = np.array([id_to_lon.get(str(pid), 0.0) for pid in self.ids], dtype=np.float32)
            else:
                print("[WARN] 未找到POI坐标文件，将基于Region构建伪地理坐标用于实验演示。")
                region_array = np.array([r[0] if len(r) > 0 else 0 for r in data['Region']], dtype=np.float32)
                cat_array = np.array([c[0] if len(c) > 0 else 0 for c in data['Catname']], dtype=np.float32)
                np.random.seed(42)
                self.lats = region_array * 0.01 + np.random.normal(0, 0.05, len(region_array)).astype(np.float32) + 40.0
                self.lons = cat_array * 0.01 + np.random.normal(0, 0.05, len(region_array)).astype(np.float32) - 74.0

            # 正弦余弦编码（仅保留此核心地理特征）
            self.geo_encodings = sinusoidal_encoding(self.lats, self.lons, pe_dim=pe_dim).astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base_vec = torch.cat([
            self.categories[idx],
            self.regions[idx],
            self.times[idx],
            self.neighbors[idx],
        ])

        if self.use_geo_features:
            geo_pe = torch.from_numpy(self.geo_encodings[idx])  # (2*pe_dim,)
            lat = torch.tensor(self.lats[idx], dtype=torch.float32)
            lon = torch.tensor(self.lons[idx], dtype=torch.float32)
            feature = torch.cat([base_vec, geo_pe])
            # 返回 (pid, feature, lat, lon)
            return self.ids[idx], feature, lat, lon
        else:
            return self.ids[idx], base_vec

    def get_input_dim(self):
        base_dim = self.cat_num + self.region_num + self.time_num + self.neighbor_num
        if self.use_geo_features:
            return base_dim + 2 * self.pe_dim
        return base_dim
