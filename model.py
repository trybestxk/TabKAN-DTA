import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from FastKANlob import FastKAN


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)

    def forward(self, query, key, value, mask=None):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, sent_len_Q, sent_len_K]

        # 处理掩码
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch_size, n_heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, sent_len_Q, n_heads, hid_dim / n_heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch_size, sent_len_Q, hid_dim]
        x = self.fc(x)
        # x = [batch_size, sent_len_Q, hid_dim]

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, sent_len, hid_dim]

        x = x.permute(0, 2, 1)
        # x = [batch_size, hid_dim, sent_len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch_size, pf_dim, sent_len]
        x = self.fc_2(x)
        # x = [batch_size, hid_dim, sent_len]
        x = x.permute(0, 2, 1)
        # x = [batch_size, sent_len, hid_dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, hid_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound_len, atom_dim]
        # src = [batch_size, protein_len, hid_dim] # encoder output
        # trg_mask = [batch_size, 1, 1, compound_len]
        # src_mask = [batch_size, 1, 1, protein_len]

        # 处理src，确保它有正确的维度
        if len(src.shape) == 2:  # [batch_size, hid_dim]
            batch_size = src.shape[0]
            src = src.unsqueeze(1)  # [batch_size, 1, hid_dim]

        # 自注意力 (self-attention)
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        # 交叉注意力 (cross-attention)
        # 注意：对于单个节点的src，我们不使用src_mask
        if src.shape[1] == 1:
            trg = self.ln(trg + self.do(self.ea(trg, src, src, None)))
        else:
            trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        # 前馈网络
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class FastKANGatedFusion(nn.Module):
    def __init__(
            self,
            hidden_dim,
            grid_min=-3.0,
            grid_max=3.0,
            num_grids=8,
            use_base_update=True,
            base_activation=F.relu,
            spline_weight_init_scale=0.01,
    ):
        super().__init__()

        # 使用完整的FastKAN网络而不是单层
        self.gate_net1 = FastKAN(
            layers_hidden=[hidden_dim, hidden_dim * 2, hidden_dim],
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )

        self.gate_net2 = FastKAN(
            layers_hidden=[hidden_dim, hidden_dim * 2, hidden_dim],
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )

        # 交互门控网络也使用多层FastKAN
        self.cross_gate = FastKAN(
            layers_hidden=[hidden_dim * 2, hidden_dim * 2, hidden_dim],
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )

    def forward(self, feat_1, feat_2):
        # 使用多层FastKAN获得更复杂的门控表示
        gate_1 = torch.sigmoid(self.gate_net1(feat_1))
        gate_2 = torch.sigmoid(self.gate_net2(feat_2))

        # 计算交互门控
        combined = torch.cat([feat_1, feat_2], dim=-1)
        combined = self.cross_gate(combined)
        cross_gate = torch.sigmoid(combined)

        # 增强的融合策略
        fused = gate_1*feat_1 + gate_2 * feat_2 + cross_gate * (combined)
        return fused


class FKCfusion(nn.Module):
    def __init__(self, hidden_dim, decoder_heads=4, dropout=0.1):
        super().__init__()

        # KAN门控融合模块
        self.drug_kan_fusion = FastKANGatedFusion(
            hidden_dim=hidden_dim,
            grid_min=-3.0,
            grid_max=3.0,
            num_grids=6,
            use_base_update=True
        )

        self.protein_kan_fusion = FastKANGatedFusion(
            hidden_dim=hidden_dim,
            grid_min=-3.0,
            grid_max=3.0,
            num_grids=6,
            use_base_update=True
        )

        # TabPFN特征提取器
        self.tabpfn_extractor = nn.Sequential(
            nn.Linear(192, 230),
            nn.ReLU(),
            nn.Linear(230, hidden_dim)
        )

        # 交叉注意力层
        self.cross_atten = DecoderLayer(
            hid_dim=hidden_dim,
            n_heads=decoder_heads,
            dropout=dropout
        )

        # 对interaction特征和tabpfn特征应用KAN门控
        self.final_kan_fusion = FastKANGatedFusion(
            hidden_dim=hidden_dim,
            grid_min=-3.0,
            grid_max=3.0,
            num_grids=6,
            use_base_update=True
        )

    def forward(self, drug_feat, protein_feat, drug_graph_feat, protein_graph_feat, tabpfn_features):
        # 使用KAN门控融合药物特征
        drug_fused = self.drug_kan_fusion(drug_feat, drug_graph_feat)

        # 使用KAN门控融合蛋白质特征
        protein_fused = self.protein_kan_fusion(protein_feat, protein_graph_feat)

        # 处理TabPFN特征
        tabpfn_feat = self.tabpfn_extractor(tabpfn_features)

        # 交叉注意力融合药物和蛋白质特征
        drug_fused_seq = drug_fused.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        protein_fused_seq = protein_fused.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        interaction_feat = self.cross_atten(drug_fused_seq, protein_fused_seq).squeeze(1)

        # 使用KAN门控融合interaction特征和tabpfn特征
        final_feat = self.final_kan_fusion(interaction_feat, tabpfn_feat)

        return final_feat


class TabKAN(nn.Module):
    """多模态药物-靶点亲和力预测模型，使用KAN进行特征融合"""

    def __init__(self,
                 tabpfn_dim=192,
                 smiles_dim=384,
                 esm_dim=1152,
                 graph_model=None,
                 graph_feat_dim=432,
                 fusion_dim=512,
                 dropout=0.1,
                 kan_grid_size=5,
                 kan_spline_order=3):
        super(TabKAN, self).__init__()

        # 保存图模型
        self.graph_model = graph_model

        # 定义维度参数
        self.dropout = dropout
        self.smiles_dim = smiles_dim
        self.esm_dim = esm_dim
        self.graph_feat_dim = graph_feat_dim
        self.decoder_heads = 4

        # 层归一化
        self.drug_ln = nn.LayerNorm(graph_feat_dim)
        self.target_ln = nn.LayerNorm(graph_feat_dim)

        # 特征转换层
        self.fc1 = nn.Linear(graph_feat_dim, graph_feat_dim)  # 处理药物图特征
        self.fc2 = nn.Linear(esm_dim, graph_feat_dim)  # 处理蛋白质嵌入
        self.fc3 = nn.Linear(smiles_dim, graph_feat_dim)  # 处理SMILES嵌入
        self.fc4 = nn.Linear(graph_feat_dim, graph_feat_dim)  # 处理蛋白质图特征

        # 融合模块
        self.fusion_module = FKCfusion(
            hidden_dim=graph_feat_dim,
            decoder_heads=self.decoder_heads,
            dropout=dropout
        )

        self.lin = nn.Sequential(
            nn.Linear(graph_feat_dim, 1024),  # 只使用融合后的特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, g, node_feats, esm_embeddings, smiles_embeddings, tabpfn_features):
        # 1. 处理图数据
        if self.graph_model is not None:
            # 使用预训练的图模型提取特征
            graph_ligand_feat, graph_protein_feat, _ = self.graph_model(g, node_feats)

        # 处理图特征
        drug_graph_feat = self.drug_ln(self.fc1(graph_ligand_feat))  # [batch_size, graph_feat_dim]
        protein_graph_feat = self.target_ln(self.fc4(graph_protein_feat))  # [batch_size, graph_feat_dim]

        # 2. 处理序列数据 - 现在是平均后的嵌入
        # 转换维度
        drug_feat = self.drug_ln(self.fc3(smiles_embeddings))  # [batch_size, graph_feat_dim]
        protein_feat = self.target_ln(self.fc2(esm_embeddings))  # [batch_size, graph_feat_dim]

        # 使用融合模块处理特征
        final_feat = self.fusion_module(
            drug_feat, protein_feat,
            drug_graph_feat, protein_graph_feat,
            tabpfn_features
        )

        # 最终预测
        prediction = self.lin(final_feat)

        return prediction


class TabKANDTA(nn.Module):
    def __init__(self,
                 tabpfn_dim=192,
                 smiles_dim=384,
                 esm_dim=1152,
                 graph_model=None,
                 graph_feat_dim=432,
                 fusion_dim=512,
                 dropout=0.1,
                 kan_grid_size=6,
                 kan_spline_order=4):
        super(TabKANDTA, self).__init__()

        self.model = TabKAN(
            tabpfn_dim=tabpfn_dim,
            smiles_dim=smiles_dim,
            esm_dim=esm_dim,
            graph_model=graph_model,
            graph_feat_dim=graph_feat_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order
        )

    def forward(self, g, node_feats, esm_embeddings, smiles_embeddings, tabpfn_features):
        return self.model(g, node_feats, esm_embeddings, smiles_embeddings, tabpfn_features)
