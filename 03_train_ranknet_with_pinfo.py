# -*- coding: utf-8 -*-
"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
"""
"""
增强版 RankNet 训练脚本：融合 He 节点特征（20维/节点） + 全局物理信息（Pinfo）

功能：
- 自动配对 caseXXX_Y.csv 和 caseXXX_Pinfo_Y.pkl
- 支持 batch 训练、早停、最佳模型保存
"""

import os
import sys
import pickle
import torch
from pathlib import Path
import datetime
import numpy as np
from learning.model import RankNetNew
from learning.utils import process_ranknet_new


def log(msg, logfile=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    if logfile is not None:
        with open(logfile, mode='a', encoding='utf-8') as f:
            f.write(line + '\n')


def load_paired_data(data_dir: str, problem: str, split: str, n_sample: int = -1):
    """
    加载配对数据：.csv（41维向量） + _Pinfo_.pkl（物理信息）

    CSV 结构（41 elements）:
      [0:18]   -> node1_vals (18)
      [18]     -> node1_depth
      [19]     -> node1_maxdepth
      [20:38]  -> node2_vals (18)
      [38]     -> node2_depth
      [39]     -> node2_maxdepth
      [40]     -> comp_res (-1 or 1)
    pkl 结构(n_vars行 * 7列)

    """
    csv_dir = Path(data_dir) / problem / f"{split}_milp_samples"
    pkl_dir = csv_dir

    csv_files = sorted(csv_dir.glob("*.csv"))
    if n_sample > 0:
        csv_files = csv_files[:n_sample]

    X_he, X_pinfo, y = [], [], []

    for csv_file in csv_files:
        stem = csv_file.stem
        parts = stem.rsplit('_', 1)
        if len(parts) < 2:
            continue
        prefix, comp_id_str = parts
        try:
            comp_id = int(comp_id_str)
        except ValueError:
            continue

        pkl_file = pkl_dir / f"{prefix}_Pinfo_{comp_id}.pkl"  # ljm debug
        if not pkl_file.exists():
            log(f"警告: Missing Pinfo file for {csv_file.name}, skipping.")
            continue

        # === 加载 CSV（41维向量）===
        try:
            f_array = np.loadtxt(csv_file, delimiter=',')
            if f_array.size != 41:
                log(f"警告: Invalid He's feature length in {csv_file} ({f_array.size}), skipping.")
                continue

            comp_res = int(f_array[40])  # -1 or 1
            label = comp_res
            y.append(label)

        except Exception as e:
            log(f"csv报错: Error loading {csv_file}: {e}")
            continue

        # === 加载 Pinfo ===  # ljm debug
        try:
            with open(pkl_file, 'rb') as f:
                pinfo = pickle.load(f)  # (n_vars, n_phys)
            if pinfo.size == 0:
                continue
        except Exception as e:
            log(f"pkl报错: Error loading {pkl_file}: {e}")
            continue
        X_pinfo.append(pinfo)  # 能提取物理信息时

        X_he.append(f_array[:40])  # shape: (40,)  # 写法1
        # X_he.append(f_array[:-1])                # 写法2
    X_he = np.array(X_he, dtype=np.float32)
    assert X_he.shape[1] == 40, f"警告: 寻找样本对过程中报错: Expected 40-dim He feature, got {X_he.shape[1]}"

    # X_pinfo = np.array([i for i in range(len(X_he))])  # ljm debug: 若没有物理信息特征,又想调试时,使用该代码

    return X_he, X_pinfo, y


def collate_batch(batch):
    """
    核心修改：
    1. 从X_he中提取node1/node2深度 → 计算深度权重（BCELoss加权项）
    2. 移除静态池化，保留原始X_pinfo（适配模型内动态池化）
    """
    X_he_list, X_pinfo_list, y_list = zip(*batch)

    # 1. 堆叠He特征（原有逻辑保留）
    X_he = torch.tensor(np.stack(X_he_list), dtype=torch.float32)  # (B, 40)
    assert X_he.shape[1] == 40, f"警告:Expected 40-dim He feature, got {X_he.shape[1]}"

    # 2. 移除静态池化：保留原始X_pinfo（list格式，每个元素是(N,7)的物理特征）
    # 注：不再做max/mean池化，交给模型内的动态注意力池化处理
    X_pinfo = [torch.tensor(pinfo, dtype=torch.float32) for pinfo in X_pinfo_list]

    # 3. 标签处理（原有逻辑保留）
    y = torch.tensor(y_list, dtype=torch.float32)  # (B,)

    # 4. 从X_he中提取深度 + 计算深度权重（BCELoss加权项）
    # 匹配你的数据结构：X_he[:,18]=node1深度，X_he[:,38]=node2深度
    depth_s = X_he[:, 18]  # node1深度（第19列，索引18）
    depth_t = X_he[:, 38]  # node2深度（第39列，索引38）

    # 严格复用你的加权公式，+1e-8避免除零错误
    abs_diff = torch.abs(depth_s - depth_t)
    min_depth = torch.min(torch.stack([depth_s, depth_t]), dim=0)[0] + 1e-8
    # sample_weight = torch.exp(  (1+abs_diff)/min_depth   )  # (B,) 每个样本的加权项 方法1
    sample_weight = (1 + abs_diff) / min_depth                # (B,) 每个样本的加权项 方法2

    # 返回值：X_he + 原始X_pinfo（无静态池化） + y + 深度权重
    return X_he, X_pinfo, y, sample_weight


## 2026.4.1 写法2: 惰性打包数据,在每个batch加载,而不是在batch之前一次性全部放进内存里
def load_paired_data_lazy(data_dir: str, problem: str, split: str, n_sample: int = -1):
    """
    惰性加载：只记录文件路径，不读取具体内容，避免内存爆炸
    """
    csv_dir = Path(data_dir) / problem / f"{split}_milp_samples"
    csv_files = sorted(csv_dir.glob("*.csv"))
    if n_sample > 0:
        csv_files = csv_files[:n_sample]

    # 这里我们只存储 (csv_path, pkl_path, label) 的元组
    # 注意：这里不再加载 numpy 数组，而是存路径字符串
    dataset_paths = []

    for csv_file in csv_files:
        stem = csv_file.stem
        parts = stem.rsplit('_', 1)
        if len(parts) < 2:
            continue
        prefix, comp_id_str = parts
        try:
            comp_id = int(comp_id_str)
        except ValueError:
            continue

        pkl_file = csv_dir / f"{prefix}_Pinfo_{comp_id}.pkl"
        if not pkl_file.exists():
            log(f"警告: Missing Pinfo file for {csv_file.name}, skipping.")
            continue

        # 读取 label (这里只读 label，因为 label 很小，占用的内存可以忽略)
        try:
            f_array = np.loadtxt(csv_file, delimiter=',')
            if f_array.size != 41: continue
            label = int(f_array[40])

            # 存储路径和标签
            dataset_paths.append((str(csv_file), str(pkl_file), label))
        except:
            continue

    return dataset_paths


## 2026.4.1 写法2: 惰性打包数据,在每个batch加载,而不是在batch之前一次性全部放进内存里
def collate_batch_lazy(batch_paths):
    """
    惰性打包函数
    Args:
        batch_paths: List[Tuple(csv_path, pkl_path, label)]
    Returns:
        X_he_batch: (Batch, 40)
        X_pinfo_list: List[Tensor]
        y_batch: (Batch,)
        sample_weight: (Batch,)  # 与写法1保持一致
    """
    X_he_list = []
    X_pinfo_list = []
    y_list = []

    for csv_path, pkl_path, label in batch_paths:
        # 1. 读取 He 特征
        try:
            f_array = np.loadtxt(csv_path, delimiter=',')
            X_he_list.append(f_array[:40])
        except Exception as e:
            print(f"报错：He等人的标签Error loading csv {csv_path}: {e}")
            continue

        # 2. 读取 Pinfo (这是大文件，在这里读)
        try:
            with open(pkl_path, 'rb') as f:
                pinfo = pickle.load(f)
            # 转换为 Tensor 并放入列表
            # 假设 pinfo 是 numpy 数组
            X_pinfo_list.append(torch.tensor(pinfo, dtype=torch.float))
        except Exception as e:
            print(f"报错：pinfo信息Error loading pkl {pkl_path}: {e}")
            # 如果 pinfo 加载失败，可以选择跳过或给一个全零的占位符
            # 这里简单起见，如果失败我们给一个空张量（需确保模型能处理空张量）
            X_pinfo_list.append(torch.tensor([], dtype=torch.float).reshape(0, 7))

        y_list.append(label)

    # 堆叠 He 特征和 Label
    X_he_batch = torch.tensor(np.array(X_he_list), dtype=torch.float)
    y_batch = torch.tensor(y_list, dtype=torch.float)

    # 完全复用写法1的sample_weight逻辑，保证一致
    depth_s = X_he_batch[:, 18]  # node1深度
    depth_t = X_he_batch[:, 38]  # node2深度
    abs_diff = torch.abs(depth_s - depth_t)
    min_depth = torch.min(torch.stack([depth_s, depth_t]), dim=0)[0] + 1e-8
    sample_weight = (1 + abs_diff) / min_depth

    return X_he_batch, X_pinfo_list, y_batch, sample_weight


def main():
    # ===== 配置 =====
    problem = "case118"
    # lr = 0.001
    lr = 0.0005
    n_epoch = 20
    n_sample = -1
    patience = 10
    early_stopping = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_train = 32
    batch_valid = 32
    data_dir = r"D:\LiJiamigFile\Comparenodes_data\data"
    log_dir = r"D:\LiJiamigFile\Comparenodes_data\log_MLP"

    loss_fn = torch.nn.BCELoss()
    optimizer_fn = torch.optim.Adam

    # ===== 命令行参数解析（略，同前）=====
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        elif sys.argv[i] == '-lr':
            lr = float(sys.argv[i + 1])
        elif sys.argv[i] == '-n_epoch':
            n_epoch = int(sys.argv[i + 1])
        elif sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
        elif sys.argv[i] == '-patience':
            patience = int(sys.argv[i + 1])
        elif sys.argv[i] == '-early_stopping':
            early_stopping = int(sys.argv[i + 1])
        elif sys.argv[i] == '-device':
            device = torch.device(sys.argv[i + 1])
        elif sys.argv[i] == '-batch_train':
            batch_train = int(sys.argv[i + 1])
        elif sys.argv[i] == '-batch_valid':
            batch_valid = int(sys.argv[i + 1])
        elif sys.argv[i] == '-data_dir':
            data_dir = str(sys.argv[i + 1])
        elif sys.argv[i] == '-log_dir':
            log_dir = str(sys.argv[i + 1])

    # ===== 日志 =====
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f'{problem}-MLP-Pinfo.txt')
    if os.path.exists(logfile):
        print(f"Error: Log file already exists: {logfile}", file=sys.stderr)
        sys.exit(1)

    ## 写法1 直接加载整个数据集到内存
    ## 备注,实测发现case1888-v1的(17k行/样本的)35k个样本10分钟都无法直接加载完毕
    # # ===== 数据加载 =====
    # log("Loading paired data...", logfile)
    # X_he_train, X_pinfo_train, y_train = load_paired_data(data_dir, problem, "train", n_sample)
    # X_he_valid, X_pinfo_valid, y_valid = load_paired_data(data_dir, problem, "valid", n_sample)
    # log(f"He Train: {len(X_he_train)} samples", logfile)
    # log(f"He Valid: {len(X_he_valid)} samples", logfile)
    # if len(X_he_train) == 0:
    #     log("No training data!", logfile)
    #     sys.exit(1)
    # train_dataset = list(zip(X_he_train, X_pinfo_train, y_train))
    # valid_dataset = list(zip(X_he_valid, X_pinfo_valid, y_valid))


    # 2026.4.1 写法2: 惰性打包数据,在每个batch加载,而不是在batch之前一次性全部放进内存里
    # ===== 数据加载 (修改处) =====
    log("Loading paired data paths (Lazy)...", logfile)
    # 现在 train_dataset 只是一个包含路径的列表，内存占用极小
    train_dataset = load_paired_data_lazy(data_dir, problem, "train", n_sample)
    valid_dataset = load_paired_data_lazy(data_dir, problem, "valid", n_sample)
    log(f"Loaded {len(train_dataset)} training pairs (Lazy Mode)", logfile)      # 此时 train_dataset 是包含路径元组的列表
    log(f"Loaded {len(valid_dataset)} validation pairs (Lazy Mode)", logfile)
    if len(train_dataset) == 0:
        log("报错: Error: No training data found! Check paths or file formats.", logfile)
        sys.exit(1)


    # # ===== 模型 =====
    # # 复现Abdel的原始Ranknet---------------------------------------------------------------------------------------------------------------------
    # n_phys = 0
    # model = RankNetNew(use_pinfo=False).to(device)

    ## 使用注意力池化参数,让物理特征参与训练------------------------------------------------------------------------------------------------------------
    # n_phys = X_pinfo_train[99].shape[1]
    # 对于case118来说,表示: 取出第100个样本的物理特征,即1个(5184,6)numpy数组,shape[1]是6
    n_phys=4  # 去掉pmax和pmin的原始数值,只保留pmax_L和pmin_L
    model = RankNetNew(use_pinfo=True, n_phys=n_phys).to(device)

    # 定义优化器
    optimizer = optimizer_fn(model.parameters(), lr=lr)
    # # 定义调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    log("-------------------------", logfile)
    log(f"RankNetNew (He:20+20, Phys:{n_phys})", logfile)
    log(f"Train He and Pinfo  on:  {len(train_dataset)} samples", logfile)
    log(f"Valid He and Pinfo  on:  {len(valid_dataset)} samples", logfile)
    log(f"Batch Size Train:        {batch_train}", logfile)
    log(f"Batch Size Valid         {batch_train}", logfile)
    log(f"Learning rate:           {lr} ", logfile)
    log(f"Number of epochs:        {n_epoch}", logfile)
    log(f"Device:                  {device}", logfile)
    log(f"Loss fct:                {loss_fn}", logfile)
    log(f"Optimizer:               {optimizer_fn}", logfile)
    log(f"Model's Size:            {sum(p.numel() for p in model.parameters())} parameters ", logfile)
    log("-------------------------", logfile)

    # ===== 训练循环 =====
    best_valid_loss = float('inf')
    patience_counter = 0
    model_save_path = Path(log_dir) / f'policy_{problem}_ranknet_pinfo.pkl'

    for epoch in range(n_epoch):
        log(f"Epoch {epoch + 1}/{n_epoch}", logfile)

        ## 写法1 直接加载整个数据集到内存
        # train_loss, train_acc = process_ranknet_new(
        #     model, train_dataset, loss_fn, device, optimizer,
        #     batch_size=batch_train, collate_fn=collate_batch  # 2种池化 14 从collate_batch函数可查询,把原始的7维池化为14维
        # )

        # 2026.4.1 写法2: 惰性打包数据,在每个batch加载,而不是在batch之前一次性全部放进内存里
        train_loss, train_acc = process_ranknet_new(
            model, train_dataset, loss_fn, device, optimizer,
            batch_size=batch_train, collate_fn=collate_batch_lazy  # 使用新的惰性打包函数
        )

        log(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}", logfile)

        ## 写法1 直接加载整个数据集到内存
        # valid_loss, valid_acc = process_ranknet_new(
        #     model, valid_dataset, loss_fn, device, optimizer=None,
        #     batch_size=batch_valid, collate_fn=collate_batch
        # )

        # 2026.4.1 写法2: 惰性打包数据,在每个batch加载,而不是在batch之前一次性全部放进内存里
        valid_loss, valid_acc = process_ranknet_new(
            model, valid_dataset, loss_fn, device, optimizer=None,
            batch_size=batch_valid, collate_fn=collate_batch_lazy  # 使用新的惰性打包函数
        )

        log(f"Valid loss: {valid_loss:.4f}, acc: {valid_acc:.4f}", logfile)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            log(f"Best model saved (loss: {valid_loss:.4f})", logfile)
        else:
            patience_counter += 1
            if patience_counter == 4:
                log(f"  4 epochs without improvement, decreasing learning rate, current lr: {scheduler.get_last_lr()[0]:.2e}", logfile)
            elif patience_counter == patience:
                log(f"  {patience} epochs without improvement, early stopping, current lr: {scheduler.get_last_lr()[0]:.2e}", logfile)
                break
            # if patience_counter >= patience:
            #     log("Early stopping", logfile)
            #     break

        # # 定义调度器
        # # 在 Epoch 循环末尾调用
        scheduler.step(valid_loss)

    log(f"Training done. Best model: {model_save_path}", logfile)


if __name__ == "__main__":
    main()