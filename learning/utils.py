"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
"""

import torch
import torch_geometric
from torch_scatter import scatter
from torch.utils.data import DataLoader



def process_ranknet_new(model, dataset, loss_fct, device, optimizer=None, batch_size=32, collate_fn=None):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(optimizer is not None),
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,             # 4个线程cpu加载训练集的原始数据
        pin_memory=True,             # 加速CPU→GPU传输
        persistent_workers=True,   # 保持worker进程，进一步加速
        # num_workers=0,            # 4个线程cpu加载训练集的原始数据
        # pin_memory=False,         # 加速CPU→GPU传输
        # persistent_workers=False  # 保持worker进程，进一步加速
    )

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    model.train() if optimizer is not None else model.eval()

    with torch.set_grad_enabled(optimizer is not None):
        for X_he, X_pinfo_list, y, sample_weight in dataloader:
            X_he = X_he.to(device)
            # 关键：X_pinfo_list是list，逐个移到device
            X_pinfo_list = [pinfo.to(device) for pinfo in X_pinfo_list]
            sample_weight = sample_weight.to(device)

            y_true = 0.5 * y + 0.5
            y_true = y_true.to(device)

            node1_feat = X_he[:, :20]
            node2_feat = X_he[:, 20:]

            # # 2026.4.1 佳明修改 让node1和node2使用的变量特征不同
            # phys_dim = X_pinfo_list[0].shape[1] - 4  # 自动推算物理特征维度
            # pinfo_node1_list = []
            # pinfo_node2_list = []
            # for pinfo in X_pinfo_list:
            #     # pinfo: (N_i, phys_dim + 4)
            #     phys_part = pinfo[:, :phys_dim]  # (N_i, phys_dim)
            #     n1_bound_part = pinfo[:, phys_dim:phys_dim + 2]  # (N_i, 2) -> N1 LB, UB
            #     n2_bound_part = pinfo[:, phys_dim + 2:phys_dim + 4]  # (N_i, 2) -> N2 LB, UB
            #     # 拼接成 Node1 的输入: (N_i, phys_dim + 2)
            #     pinfo_node1_list.append(torch.cat([phys_part, n1_bound_part], dim=1))
            #     # 拼接成 Node2 的输入: (N_i, phys_dim + 2)
            #     pinfo_node2_list.append(torch.cat([phys_part, n2_bound_part], dim=1))

            # 2026.4.10 佳明修改 pinfo由6列改为4列
            pinfo_node1_list = []
            pinfo_node2_list = []
            for pinfo in X_pinfo_list:
                # pinfo: (N_i, Total_Features)

                # 1. 先切片取出我们想要的部分（跳过前2列，取到倒数第4列之前）
                # [:, 2:-4] 意思是：从索引2开始，直到倒数第4个元素结束（不包含倒数第4个）
                phys_part = pinfo[:, 2:-4]

                # 2. 直接根据切出来的结果计算维度，不再依赖魔法数字减法
                current_phys_dim = phys_part.shape[1]

                # 3. 提取边界部分（依然是在物理特征之后的4列）
                # 起始索引 = 2 (跳过的) + current_phys_dim
                start_idx = 2 + current_phys_dim

                n1_bound_part = pinfo[:, start_idx: start_idx + 2]
                n2_bound_part = pinfo[:, start_idx + 2: start_idx + 4]

                # 拼接
                pinfo_node1_list.append(torch.cat([phys_part, n1_bound_part], dim=1))
                pinfo_node2_list.append(torch.cat([phys_part, n2_bound_part], dim=1))

            # 修改模型 forward 的签名，让它接受两个列表
            y_proba = model(node1_feat, node2_feat, pinfo_node1_list, pinfo_node2_list)


            ## 2026.3.7 旧版代码 node1和node2使用的变量特征完全相同
            # # 传入原始X_pinfo_list，模型内做池化
            # y_proba = model(node1_feat, node2_feat, X_pinfo_list)


            ## 不带节点深度加权
            weighted_loss = loss_fct(y_proba, y_true)
            ## 带节点深度加权
            # loss_fct.reduction = "none"
            # raw_loss = loss_fct(y_proba, y_true)
            # weighted_loss = (raw_loss * sample_weight).mean()

            if optimizer is not None:
                optimizer.zero_grad()
                weighted_loss.backward()

                # 梯度裁剪：防止梯度爆炸 (解决之前日志中 Loss 飙升的关键)
                # 在 optimizer.step() 之前添加
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            acc = ((y_proba > 0.5) == y_true).float().mean().item()
            total_loss += weighted_loss.item()
            total_acc += acc
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches