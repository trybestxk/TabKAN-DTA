import torch
import dgl
import numpy as np


class MultiModalDataLoader:
    """多模态数据加载器，整合图数据、蛋白质嵌入、药物嵌入和表格特征"""

    def __init__(self,
                 graph_dataset,
                 protein_embed_path=None,
                 smiles_embed_path=None,
                 tabpfn_features=None,
                 batch_size=64,
                 shuffle=False):
        self.graph_dataset = graph_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 加载蛋白质嵌入 (现在是平均后的嵌入)
        if protein_embed_path:
            print(f"加载蛋白质嵌入: {protein_embed_path}")
            self.protein_embeddings = torch.load(protein_embed_path)
            self.protein_embed_dim = self.protein_embeddings.shape[1]  # 从嵌入维度确认
            print(f"蛋白质嵌入形状: {self.protein_embeddings.shape}")
        else:
            print("未提供蛋白质嵌入路径")
            raise ValueError("必须提供蛋白质嵌入路径")

        # 加载SMILES嵌入 (现在是平均后的嵌入)
        if smiles_embed_path:
            print(f"加载SMILES嵌入: {smiles_embed_path}")
            self.smiles_embeddings = torch.load(smiles_embed_path)
            self.smiles_dim = self.smiles_embeddings.shape[1]  # 从嵌入维度确认
            print(f"SMILES嵌入形状: {self.smiles_embeddings.shape}")
        else:
            print("未提供SMILES嵌入路径")
            raise ValueError("必须提供SMILES嵌入路径")

        # 加载表格特征
        if tabpfn_features is not None:
            self.tabpfn_features = tabpfn_features
            print(f"TabPFN特征形状: {self.tabpfn_features.shape}")
        else:
            print("未提供TabPFN特征")
            raise ValueError("必须提供TabPFN特征")

        # 创建批次索引
        self.indices = list(range(len(graph_dataset)))
        if shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration

        # 获取当前批次的索引
        end_idx = min(self.current_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_idx:end_idx]
        self.current_idx = end_idx

        # 获取图和标签
        batch_graphs = [self.graph_dataset[i][0] for i in batch_indices]
        batch_labels = torch.tensor([self.graph_dataset[i][1] for i in batch_indices], dtype=torch.float32)

        # 批量处理图
        batched_graph = dgl.batch(batch_graphs)

        # 获取平均后的蛋白质嵌入
        batch_proteins = self.protein_embeddings[batch_indices]

        # 获取平均后的SMILES嵌入
        batch_smiles = self.smiles_embeddings[batch_indices]

        # 处理TabPFN特征
        batch_tabpfn = torch.tensor(self.tabpfn_features[batch_indices], dtype=torch.float32)

        # 返回批次数据
        return batched_graph, batch_proteins, batch_smiles, batch_tabpfn, batch_labels

    def __len__(self):
        return (len(self.graph_dataset) + self.batch_size - 1) // self.batch_size