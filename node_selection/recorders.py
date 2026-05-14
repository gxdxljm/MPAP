#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
Contains utilities to save and load comparaison behavioural data
"""


import os
import importlib.util  # 兼容性修改
import torch
import numpy as np
import re
import time
import pickle
from typing import Optional, Dict

from pyscipopt import Model, Variable


class CompFeaturizerSVM():
    def __init__(
            self,
            model: Model,
            save_dir: Optional[str] = None,
            instance_name: Optional[str] = None,
            var_info_csv: Optional[str] = None,
            var_dict: Optional[Dict[str, Variable]] = None
    ):
        """
        初始化特征记录器。

        Parameters:
            model: SCIP 模型实例
            save_dir: 保存目录
            instance_name: 实例名称（用于文件命名）
            var_info_csv: 外部 CSV 路径（第1列变量名，后续列为物理特征）
            var_dict: {变量名: Variable} 映射字典（必须由外部提供）
            2026.4.3修正, var_dict: {变量名: Variable} 映射字典（无需外部提供）
        """
        self.instance_name = instance_name
        self.save_dir = save_dir
        self.m = model
        # self.var_dict = var_dict or {}                    # 所有变量的字典。   注意, 外部必须以 transformed空间构建名称, 例如: t_uit(1_10)
        # self.var_dict = self.m.getVars(transformed=True)  # 所有变量的字典。 注意, 请在求解进程开始后创建,否则变量名称不正确
        self.var_dict = {}

        # 加载外部物理信息
        if var_info_csv and os.path.exists(var_info_csv):
            self._load_var_info(var_info_csv)
        else:
            self.var_names = []
            self.phys_features = np.array([])

    def _load_var_info(self, csv_path: str):
        data = np.genfromtxt(csv_path, delimiter=',', dtype=str, skip_header=1)
        if data.size == 0:
            self.var_names = []
            self.phys_features = np.array([]).reshape(0, 0)
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.var_names = data[:, 0].tolist()  # 所有0-1变量的列表。 注意, 外部必须以 transformed空间构建csv物理文件的第1列, 例如: t_uit(1_10)
        self.phys_features = data[:, 1:].astype(float)


    ## 2026.4.1 获取4列动态的节点特征[lb_1, ub_1, lb2, ub2]
    # 版本2: getDomchg (Domain Changes)：记录的是操作。即“从父节点到当前节点，我们做了什么分支决策”。它侧重于树的拓扑结构。
    # 版本 2 只能记录 父节点（Parent Node） 的边界（即 Global 边界） + 当前节点 的分支变化（DomChg; 缺失的部分：从“根节点”到“父节点”之间发生的传播（Propagation）
    def _get_var_local_info(self, node):
        """
        利用 node.getDomchg() 和 BoundChange 列表获取节点的局部边界。
        逻辑：局部边界 = 全局边界 + 当前节点的变化(DomChg)
        """
        if not self.var_names:
            return np.array([])

        # 0. 由于该方法在求解进程中被调用,故创建当前模型的名称列表,才能获得transform空间的名称
        varlist = self.m.getVars(transformed=True)
        self.var_dict = {var.name: var for var in varlist}

        # 1. 准备一个字典来存储当前节点的变化
        # 格式: { var_name: {'lb': value, 'ub': value} }
        changes_map = {}

        # 2. 获取该节点的所有边界变化
        domchg = node.getDomchg()  # pyscipopt标准函数:https://pyscipopt.readthedocs.io/en/latest/api/node.html#pyscipopt.scip.Node.getDomchg

        if domchg is not None:
            try:
                # 获取所有变化的列表
                bound_changes = domchg.getBoundchgs()  # pyscipopt标准函数, 可在scip.pxi中的DomainChanges类下找到".getBoundchgs()"方法

                # ljm debug
                # n_bound_changes = len(bound_changes)
                # print(f"当前节点域变化的变量数: {n_bound_changes}")  # 数量为1,增量记录,只记录了父节点到当前节点的分支导致的变量边界变化

                for bc in bound_changes:
                    # 获取发生变化的变量对象
                    var = bc.getVar()
                    # 注意：这里使用 var.name 来匹配，确保和你的 var_dict 键一致
                    var_name = var.name

                    # print(f"由父节点到当前节点的分支变量: {var_name}")

                    if var_name not in changes_map:
                        changes_map[var_name] = {'lb': None, 'ub': None}

                    # 获取新值
                    new_val = bc.getNewBound()  # pyscipopt标准函数, 可在scip.pxi中的BoundChange类下找到".getNewBound()"方法

                    # 判断是上界还是下界
                    # 0 = lower, 1 = upper (根据 getBoundtype 的文档)
                    bound_type = bc.getBoundtype()  # pyscipopt标准函数, 可在scip.pxi中的BoundChange类下找到".getBoundtype()"方法

                    if bound_type == 0:  # Lower Bound
                        changes_map[var_name]['lb'] = new_val
                    else:  # Upper Bound
                        changes_map[var_name]['ub'] = new_val

            except Exception as e:
                print(f"报错: Error processing bound changes: {e}")

        # 3. 结合全局边界计算最终局部边界
        local_bounds = []
        for var_name in self.var_names:
            var = self.var_dict.get(var_name)
            if var is None:
                lb_local = ub_local = np.nan
            else:
                # 默认使用SCIP预求解后识别出来的全局边界
                lb_local = var.getLbGlobal()
                ub_local = var.getUbGlobal()

                # 如果该变量在当前节点有变化，更新边界
                # 注意：DomChg 只包含相对于父节点的变化，所以这里我们直接覆盖
                # 因为全局边界是基础，DomChg 是在此基础上的收紧
                if var_name in changes_map:
                    if changes_map[var_name]['lb'] is not None:

                        # ljm debug
                        # print(f"由父节点到当前节点的分支变量: {var_name}, lb_old: {lb_local}, lb_new: {changes_map[var_name]['lb']}")

                        lb_local = changes_map[var_name]['lb']
                    if changes_map[var_name]['ub'] is not None:

                        # ljm debug
                        # print(f"由父节点到当前节点的分支变量: {var_name}, ub_old: {ub_local}, ub_new: {changes_map[var_name]['ub']}")

                        ub_local = changes_map[var_name]['ub']


            local_bounds.append([lb_local, ub_local])

        return np.array(local_bounds)


    def _sanitize_array(self, arr: np.ndarray) -> np.ndarray:
        """清洗 nan/inf 为 MLP 友好数值"""
        arr = np.where(np.isnan(arr), 0.0, arr)
        arr = np.where(np.isposinf(arr), 1e6, arr)
        arr = np.where(np.isneginf(arr), -1e6, arr)
        return arr


    def save_comp(self, model, node1, node2, comp_res, comp_id):

        # He等人的特征和标签,保存为.csv
        f1, f2 = self.get_features(node1), self.get_features(node2)
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.csv")
        file = open(file_path, 'a')
        np.savetxt(file, f1, delimiter=',')
        np.savetxt(file, f2, delimiter=',')
        file.write(str(comp_res))
        file.close()

        # 携带物理信息的全局变量特征 & 局部边界处理
        if hasattr(self, 'var_names') and self.var_names:
            phys = self.phys_features

            # --- 新增逻辑：分别获取两个节点的局部边界 ---
            # 获取 Node 1 的局部边界 [n_vars, 2]
            node1_bounds = self._get_var_local_info(node1)

            # 获取 Node 2 的局部边界 [n_vars, 2]
            node2_bounds = self._get_var_local_info(node2)

            ## ljm debug
            # node_diff = np.abs(node1_bounds - node2_bounds)
            # node_diff_indices = np.where(node_diff > 1e-5)[0]
            # print(f"共有 {len(self.var_names)} 个变量, 发现 {len(node_diff_indices)} 个变量边界不同, 分别是 {node_diff_indices}")


            # --- 核心步骤：水平拼接 ---
            # 确保获取到的边界数据不为空
            if phys.size > 0 and node1_bounds.size > 0 and node2_bounds.size > 0:

                # 直接将 [物理特征, Node1边界, Node2边界] 拼在一起
                # 最终形状: (n_variables, n_phys_features + 2 + 2)
                final_features = np.concatenate([phys, node1_bounds, node2_bounds], axis=1)

                # 清洗数据
                final_features = self._sanitize_array(final_features)

                # 保存文件
                pkl_path = os.path.join(
                    self.save_dir,
                    # f"{self.instance_name}_Pinfo_With_LocalBounds_{comp_id}.pkl"
                    f"{self.instance_name}_Pinfo_{comp_id}.pkl"
                )
                with open(pkl_path, 'wb') as f:
                    pickle.dump(final_features, f)

        return self


    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    def get_features(self, node):

        model = self.m

        f = []
        feat = node.getHeHeaumeEisnerFeatures(model, model.getDepth() + 1)  # pyscipopt接口函数: 提取He等人的节点特征

        for k in ['vals', 'depth', 'maxdepth']:
            if k == 'vals':

                for i in range(1, 19):
                    try:
                        f.append(feat[k][i])
                    except:
                        f.append(0)

            else:
                f.append(feat[k])  # 节点深度,请查看getHeHeaumeEisnerFeatures函数的定义.其中,model.getDepth()+1是最大深度

        return f


