"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class RankNet(torch.nn.Module):

    def __init__(self):
        super(RankNet, self).__init__()

        self.linear1 = torch.nn.Linear(20, 50)
        self.activation = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(50, 1)
        
    def forward_node(self, n):
        x = self.linear1(n)
        x = self.activation(x)
        x = self.linear2(x)
        return x
        

    def forward(self, n0,n1):
        s0,s1 = self.forward_node(n0), self.forward_node(n1)
   
        return torch.sigmoid(-s0 + s1)  # (-s0+s1)属于实数, sigmoid将实数映射到(0,1)区间,这正是BCEloss所要求的


class RankNetNew(nn.Module):
    """
    支持两种模式：
      1. use_pinfo=False → 完全等价于原始 RankNet（用于验证 baseline）
      2. use_pinfo=True  → 使用注意力池化处理原始 (N,6) 物理特征

    注意：X_pinfo 来自 collate_batch，是 list[Tensor]，每个 shape=(N,6)
    """

    def __init__(self, use_pinfo: bool = False, n_phys: int = 6):
        super().__init__()
        self.use_pinfo = use_pinfo
        self.n_phys = n_phys

        # === 核心：节点打分头（复刻 RankNet）===
        self.node_scorer = nn.Sequential(           # case118用这个
            nn.Linear(20, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )
        # self.node_scorer = nn.Sequential(
        #     nn.Linear(20, 128),   # case1888用这个
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 1)
        # )

        if self.use_pinfo:
            # === Pinfo 编码器：将每个节点的 7 维 → 32 维 ===
            self.pinfo_encoder = nn.Sequential(
                nn.Linear(n_phys, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32)
            )
            # === 注意力权重预测器 ===
            self.attention_net = nn.Sequential(
                nn.Linear(32, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1)
            )
            # === 全局特征融合到节点打分中 ===
            # 将全局信息（32维）拼接到节点特征前，再送入打分头
            self.fused_scorer = nn.Sequential(
                nn.Linear(20 + 32, 50),   # 20 (node) + 32 (global)
                nn.LeakyReLU(),
                nn.Linear(50, 1)
            )
        else:
            self.pinfo_encoder = None
            self.attention_net = None
            self.fused_scorer = None

    def attention_pooling(self, pinfo_list):
        """
        对 list[(N,6)] 做注意力池化 → (B, 32)
        """
        pooled = []
        for pinfo in pinfo_list:  # pinfo: (N, 6)
            if pinfo.size(0) == 0:
                # 安全兜底: 空 pinfo → 零向量
                pooled.append(torch.zeros(32, device=pinfo.device))
                continue
            # (N,6) → (N,32)
            emb = self.pinfo_encoder(pinfo)  # (N,32)
            # (N,32) → (N,1)
            attn_logits = self.attention_net(emb)  # (N,1)
            # Softmax over nodes
            attn_weights = torch.softmax(attn_logits, dim=0)  # (N,1)
            # Weighted sum → (32,)
            global_feat = (emb * attn_weights).sum(dim=0)  # (32,)
            pooled.append(global_feat)
        return torch.stack(pooled, dim=0)  # (B, 32)

    def forward(self, node1_feat, node2_feat, pinfo_node1_list=None, pinfo_node2_list=None):
        """
        Args:
            node1_feat: (B, 20)
            node2_feat: (B, 20)
            pinfo_node1_list: List[Tensor], each (N_i, phys_dim + 2)
            pinfo_node2_list: List[Tensor], each (N_i, phys_dim + 2)
        Returns:
            prob: (B,)
        """
        if self.use_pinfo and pinfo_node1_list is not None:
            # 1. 分别池化
            global_feat_1 = self.attention_pooling(pinfo_node1_list)  # (B, 32)
            global_feat_2 = self.attention_pooling(pinfo_node2_list)  # (B, 32)

            # 2. 融合
            node1_fused = torch.cat([node1_feat, global_feat_1], dim=1)
            node2_fused = torch.cat([node2_feat, global_feat_2], dim=1)

            # 3. 打分
            s0 = self.fused_scorer(node1_fused).squeeze(-1)
            s1 = self.fused_scorer(node2_fused).squeeze(-1)
        else:
            # 原始路径
            s0 = self.node_scorer(node1_feat).squeeze(-1)
            s1 = self.node_scorer(node2_feat).squeeze(-1)

        prob = torch.sigmoid(-s0 + s1)
        return prob

    
