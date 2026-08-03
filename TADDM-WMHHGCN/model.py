


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GINConv, global_max_pool as gmp
from Utils.utils_ import reset

EPS = 1e-15


# =====================================================
# 1️⃣ HgnnEncoder: 多头 + 残差 + LayerNorm 版本
# =====================================================
class HgnnEncoder(nn.Module):
    def __init__(self, in_channels, dim_1, num_heads=4):
        super(HgnnEncoder, self).__init__()
        self.heads = nn.ModuleList([
            HypergraphConv(in_channels, dim_1) for _ in range(num_heads)
        ])
        self.norm = nn.LayerNorm(dim_1 * num_heads)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, edge, edge_weight=None):
        head_outs = []
        for conv in self.heads:
            h = torch.relu(conv(x, edge, hyperedge_weight=edge_weight))
            head_outs.append(h)
        h_cat = torch.cat(head_outs, dim=1)
        h_cat = self.dropout(self.norm(h_cat))
        return h_cat





# =====================================================
# 3️⃣ Decoder: GELU 激活 + 双层输出头
# =====================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels      # 单节点嵌入维度（例如 2048）
        self.fc1 = None                     # 延迟初始化
        self.fc2 = None
        self.fc3 = None
        self.short = None
        self.output = None
        self.dropout = nn.Dropout(0.25)

    def _init_layers(self, h0_dim):
        """首次调用 forward 时，根据实际拼接后的维度初始化层"""
        print(f"[⚙️ Decoder auto-init] Detected input dim = {h0_dim}")

        self.fc1 = nn.Sequential(
            nn.Linear(h0_dim, h0_dim // 2),
            nn.LayerNorm(h0_dim // 2),
            nn.GELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(h0_dim // 2, h0_dim // 4),
            nn.LayerNorm(h0_dim // 4),
            nn.GELU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(h0_dim // 4, h0_dim // 8),
            nn.LayerNorm(h0_dim // 8),
            nn.GELU()
        )
        self.short = nn.Linear(h0_dim, h0_dim // 8)
        self.output = nn.Sequential(
            nn.Linear(h0_dim // 8, h0_dim // 16),
            nn.ReLU(),
            nn.Linear(h0_dim // 16, 1)
        )

        # 迁移到设备
        self.fc1, self.fc2, self.fc3, self.short, self.output = \
            self.fc1.to(next(self.parameters()).device), \
            self.fc2.to(next(self.parameters()).device), \
            self.fc3.to(next(self.parameters()).device), \
            self.short.to(next(self.parameters()).device), \
            self.output.to(next(self.parameters()).device)

    def forward(self, graph_embed, drug_id, mic_id, dis_id):
        # 拼接三元关系向量
        h_0 = torch.cat((graph_embed[drug_id, :],
                         graph_embed[mic_id, :],
                         graph_embed[dis_id, :]), dim=1)

        # ⚙️ 动态初始化（只会触发一次）
        if self.fc1 is None:
            self._init_layers(h_0.shape[1])

        # 前向传播
        h_1 = self.dropout(self.fc1(h_0))
        h_2 = self.dropout(self.fc2(h_1))
        h_3 = self.dropout(self.fc3(h_2))
        h_short = self.short(h_0)
        h_agg = torch.relu(h_3 + h_short)
        out = self.output(h_agg)

        return h_agg, out.squeeze(dim=1)


# =====================================================
# 4️⃣ HGNN 主体: Attention Summary + 动态 α
# =====================================================
class HGNN(nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder, out_channels):
        super(HGNN, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder

        self.weight = None  # 延迟初始化

        self.summary_drop = nn.Dropout(0.2)



    def attention_summary(self, z):
        # z: [N, D_z]
        D_z = z.size(1)

        # 初始化 DGI 权重（如果还没初始化）
        if self.weight is None:
            print(f"[⚙️ Auto-init self.weight in attention_summary: {D_z}×{D_z}]")
            self.weight = nn.Parameter(torch.Tensor(D_z, D_z).to(z.device))
            nn.init.xavier_uniform_(self.weight)
        elif self.weight.size(0) != D_z:
            print(f"[⚙️ Auto-adjust self.weight in attention_summary: {self.weight.size(0)} → {D_z}]")
            self.weight = nn.Parameter(torch.Tensor(D_z, D_z).to(z.device))
            nn.init.xavier_uniform_(self.weight)

        # 注意力加权求 summary
        att = torch.softmax(torch.matmul(z, self.weight.mean(1)), dim=0)
        summary = torch.sum(att.unsqueeze(1) * z, dim=0)
        return torch.sigmoid(self.summary_drop(summary))

    def _prep_summary(self, summary):
        if isinstance(summary, list):
            summary = torch.stack(summary).mean(dim=0)
        if summary.dim() == 2 and summary.size(0) > 1:
            summary = summary.mean(dim=0)
        if summary.dim() == 1:
            summary = summary.unsqueeze(1)
        return summary

    def _prep_z(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        elif z.dim() == 0:
            z = z.view(1, 1)
        return z

    def discriminate(self, z, summary, sigmoid=True):
        z = self._prep_z(z)
        summary = self._prep_summary(summary)
        D = z.size(1)

        # 动态初始化判别矩阵
        if self.weight is None or self.weight.size(0) != D:
            print(f"[⚙️ HGNN auto-init DGI weight: {D}×{D}]")
            self.weight = nn.Parameter(torch.Tensor(D, D).to(z.device))
            nn.init.xavier_uniform_(self.weight)

        s = torch.matmul(self.weight, summary)
        value = torch.matmul(z, s)
        return torch.sigmoid(value) if sigmoid else value



    def _unpack_edge(self, edge_pack):
        if isinstance(edge_pack, (tuple, list)):
            if len(edge_pack) >= 2:
                return edge_pack[0], edge_pack[1]
            else:
                return edge_pack[0], None
        else:
            return edge_pack, None

    def forward(self, dru_feature, mic_feature, dis_feature,
                edge_pos, drug_id, mic_id, dis_id, neg_edges=None):
        x_dru, x_mic, x_dis = self.bio_encoder(dru_feature, mic_feature, dis_feature)
        embed = torch.cat((x_dru, x_mic, x_dis), dim=0)

        edge_index_pos, edge_weight_pos = self._unpack_edge(edge_pos)
        graph_embed_pos = self.graph_encoder(embed, edge_index_pos, edge_weight_pos)


        emb, res = self.decoder(graph_embed_pos, drug_id, mic_id, dis_id)
        return emb, res,graph_embed_pos
class BioEncoder_wo_BioAttributes(nn.Module):
    def __init__(self,
                 drug_num: int,
                 mirna_num: int,
                 disease_num: int,
                 output: int,
                 drop_p: float = 0.3):
        """
        完全去掉分子图和相似性矩阵，仅使用 one-hot 编码。
        """
        super(BioEncoder_wo_BioAttributes, self).__init__()
        self.drug_num = drug_num
        self.mirna_num = mirna_num
        self.disease_num = disease_num

        # 各类型实体的 one-hot 投影层
        self.drug_layer = nn.Sequential(
            nn.Linear(drug_num, output),
            nn.LayerNorm(output),
            nn.ReLU(),
            nn.Dropout(drop_p)
        )
        self.mic_layer = nn.Sequential(
            nn.Linear(mirna_num, output),
            nn.LayerNorm(output),
            nn.ReLU(),
            nn.Dropout(drop_p)
        )
        self.dis_layer = nn.Sequential(
            nn.Linear(disease_num, output),
            nn.LayerNorm(output),
            nn.ReLU(),
            nn.Dropout(drop_p)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                dru_adj,
                mic_feature,
                dis_feature,
                ):
        """
        """
        device = next(self.parameters()).device

        # 直接生成 One-hot
        x_d = self.drug_layer(dru_adj)
        x_mic = self.mic_layer(mic_feature)
        x_dis = self.dis_layer(dis_feature)

        return x_d, x_mic, x_dis
