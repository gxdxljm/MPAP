# -*- coding: utf-8 -*-
"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
"""

import torch
import time
import numpy as np
from pyscipopt import Nodesel
from learning.model import RankNet, RankNetNew


class CustomNodeSelector(Nodesel):

    def __init__(self, sel_policy='', comp_policy=''):
        self.sel_policy = sel_policy
        self.comp_policy = comp_policy
        self.sel_counter = 0
        self.comp_counter = 0

        
    def nodeselect(self):
        
        self.sel_counter += 1
        policy = self.sel_policy
        
        if policy == 'estimate':
            res = self.estimate_nodeselect()
        elif policy == 'dfs':
            res = self.dfs_nodeselect()
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodeselect()
        elif policy == 'bfs':
            res = self.bfs_nodeselect()
        elif policy == 'random':
            res = self.random_nodeselect()
        else:
            res = {"selnode": self.model.getBestNode()}  # 调用scip的SCIPgetBestNode函数:https://scipopt.org/doc/html/scip__tree_8c_source.php
            # print(f"ljm debug, node_selection rule: SCIPgetBestNode")  # ljm debug
        return res
    
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        policy = self.comp_policy
        
        if policy == 'estimate':
            res = self.estimate_nodecomp(node1, node2)
        elif policy == 'dfs':
            res = self.dfs_nodecomp(node1, node2)
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodecomp(node1, node2)
        elif policy == 'bfs':
            res = self.bfs_nodecomp(node1, node2)
        elif policy == 'random':
            res = self.random_nodecomp(node1, node2)
        else:
            res = 0
            
        return res
    
    #BFS
    def bfs_nodeselect(self):
        return {'selnode':self.model.getBfsSelNode() }       # pyscipopt接口:getBfsSelNode函数,实现最佳优先搜索的节点选择功能

    ## 因为Abdel等人没有写,佳明补充,参考scip官网:https://scipopt.org/doc/html/nodesel__bfs_8c_source.php
    def bfs_nodecomp(self, node1, node2):
        """
        Implements SCIP's built-in BFS (Best-First Search) node comparison logic.
        Corresponds to `nodeselCompBfs` in SCIP source.
        """
        # Step 1: Compare by lower bound
        lb1 = node1.getLowerbound()
        lb2 = node2.getLowerbound()

        if self.model.isLT(lb1, lb2):
            return -1
        elif self.model.isGT(lb1, lb2):
            return 1
        else:
            # Step 2: Lower bounds are equal → compare by estimate
            est1 = node1.getEstimate()
            est2 = node2.getEstimate()

            # Check if both estimates are +/- infinity or equal
            both_inf_pos = self.model.isInfinity(est1) and self.model.isInfinity(est2)
            both_inf_neg = self.model.isInfinity(-est1) and self.model.isInfinity(-est2)
            est_equal = self.model.isEQ(est1, est2)

            if both_inf_pos or both_inf_neg or est_equal:
                # Step 3: Compare by node type
                type1 = node1.getType()
                type2 = node2.getType()
                CHILD = 3  # SCIP_NODETYPE_CHILD
                SIBLING = 2  # SCIP_NODETYPE_SIBLING  # 参考scip官方对定义的枚举SCIP_NodeType:https://scipopt.org/doc/html/type__tree_8h.php#afbf33e0ad72332ab6c0c00daaab7527d

                if type1 == CHILD and type2 != CHILD:
                    return -1
                elif type1 != CHILD and type2 == CHILD:
                    return 1
                elif type1 == SIBLING and type2 != SIBLING:
                    return -1
                elif type1 != SIBLING and type2 == SIBLING:
                    return 1
                else:
                    # Step 4: Compare by depth (shallower is better)
                    depth1 = node1.getDepth()
                    depth2 = node2.getDepth()
                    if depth1 < depth2:
                        return -1
                    elif depth1 > depth2:
                        return 1
                    else:
                        return 0  # completely equal

            # Step 2b: Estimates are not equal → compare them
            if self.model.isLT(est1, est2):
                return -1
            else:
                # By SCIP assertion: must be GT (since EQ handled above)
                return 1
        
    #Estimate 
    def estimate_nodeselect(self):
        return {'selnode':self.model.getEstimateSelNode() }  # pyscipopt接口:getEstimateSelNode函数,实现最佳估计搜索的节点选择功能
    
    def estimate_nodecomp(self, node1,node2):
        
        #estimate 
        estimate1 = node1.getEstimate()
        estimate2 = node2.getEstimate()
        if (self.model.isInfinity(estimate1) and self.model.isInfinity(estimate2)) or \
            (self.model.isInfinity(-estimate1) and self.model.isInfinity(-estimate2)) or \
            self.model.isEQ(estimate1, estimate2):
                lb1 = node1.getLowerbound()
                lb2 = node2.getLowerbound()
                
                if self.model.isLT(lb1, lb2):
                    return -1
                elif self.model.isGT(lb1, lb2):
                    return 1
                else:
                    ntype1 = node1.getType()
                    ntype2 = node2.getType()
                    CHILD, SIBLING = 3,2
                    
                    if (ntype1 == CHILD and ntype2 != CHILD) or (ntype1 == SIBLING and ntype2 != SIBLING):
                        return -1
                    elif (ntype1 != CHILD and ntype2 == CHILD) or (ntype1 != SIBLING and ntype2 == SIBLING):
                        return 1
                    else:
                        return -self.dfs_nodecomp(node1, node2)
     
        
        elif self.model.isLT(estimate1, estimate2):
            return -1
        else:
            return 1
        
        
        
    # Depth first search        
    def dfs_nodeselect(self):
        
        selnode = self.model.getPrioChild()  #aka best child of current node
        if selnode == None:
            
            selnode = self.model.getPrioSibling() #if current node is a leaf, get 
            # a sibling
            if selnode == None: #if no sibling, just get a leaf
                selnode = self.model.getBestLeaf()
                
        return {"selnode": selnode}
    
    def dfs_nodecomp(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    
    
    
    # Breath first search
    def breadthfirst_nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
    
    def breadthfirst_nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
     
     #random
    def random_nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    def random_nodecomp(self, node1,node2):
        return -1 if np.random.rand() < 0.5 else 1


class OracleNodeSelectorAbdel(CustomNodeSelector):

    def __init__(self, oracle_type, optsol=0, prune_policy='estimate', inv_proba=0, sel_policy=''):

        super().__init__(sel_policy=sel_policy)
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
        self.inv_proba = inv_proba
        self.sel_policy = sel_policy
        self.inf_counter  = 0

    def nodecomp(self, node1, node2, return_type=False):  # 2026.3.4 佳明注释 按时不理解res生成的过程...
        
        self.comp_counter += 1
        
        if self.oracle_type == "optimal_plunger":            
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)  # 2026.3.4 佳明注释 需要查看pyscipopt接口:getAncestorBranchings函数的功能
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            inv = np.random.rand() < self.inv_proba
            # print(f"ljm debug: 观察采样的标签是否被随机扰动, inv: {bool(inv)}")
            
            if d1 and d2:  # 2026.3.4 佳明注释 按时不理解res生成的过程...  如果都被分支过了
                res, comp_type = self.dfs_nodecomp(node1, node2), 0
            elif d1:       # 2026.3.4 佳明注释 按时不理解res生成的过程...  如果只有1被分支过了
                res = comp_type = -1
                self.inf_counter += 1
            elif d2:       # 2026.3.4 佳明注释 按时不理解res生成的过程...  如果只有2被分支过了
                res = comp_type = 1
                self.inf_counter += 1
            else:          # 2026.3.4 佳明注释 按时不理解res生成的过程...  如果 都没有 被分支过
                res, comp_type = self.estimate_nodecomp(node1, node2), 10              
            
            inv_res = -1 if res == 1 else 1
            res = inv_res if inv else res   # # 2026.3.4 佳明注释 按时不理解res生成的过程...  进行概率扰动?
            
            return res if not return_type  else  (res, comp_type)
        else:
            raise NotImplementedError

    
    def is_sol_in_domaine(self, sol, node):
        #By partionionning, it is sufficient to only check what variable have
        #been branched and if sol is in [lb, up]_v for v a branched variable
        # 通过划分，我们只需检查哪些变量被划分了，以及解 sol 是否处于 [下限，上限]_v 范围内（其中 v 是被划分的变量）即可。
        
        bvars, bbounds, btypes = node.getAncestorBranchings()

        # ljm debug
        ## 2026.4.5 经过调试发现, "t_..."格式的变量可以与"..."格式的变量互相调用,不存在无法识别的问题
        # print(f"bvars: {bvars}, bbounds: {bbounds}, btypes: {btypes}")
        ## 最后,可单独运行case118_1.lp的.sol文件和收集好的node样本，看第41行的结果、物理信息csv的lb和ub列的区别来验证node1和node2的标签是否正确
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            if btype == 0:#LOWER BOUND
                if sol[bvar] < bbound:
                    return False
            else: #btype==1:#UPPER BOUND
                if sol[bvar] > bbound:
                    return False
        
        return True
    # =================================================================
    # ljm逻辑验证笔记: 关于最优解变量缺失与分支判断的正确性
    # =================================================================
    # 1. 逻辑推演验证（假设变量名为 xi）
    # -------------------------------
    # 情况A（解为0）：
    # 最优解中分支变量 xi=0.0（sol文件中无记录，默认取0），
    # 此时 Node1: xi≤0；Node2: xi≥1。
    # 对于 Node1：btype=1 (UPPER), bbound=0。执行 else: if sol[xi]>bbound。
    # 因 0.0 > 0 不成立，故 d1=True（包含最优解）。
    # 对于 Node2：btype=0 (LOWER), bbound=1。执行 if: if sol[xi]<bbound。
    # 因 0.0 < 1 成立，故 d2=False（不包含最优解）。
    # 结论：Node1 胜出，逻辑正确。
    #
    # 情况B（解为1）：
    # 最优解中分支变量 xi=1.0（sol文件中有记录，值为1），
    # 此时 Node1: xi≤0；Node2: xi≥1。
    # 对于 Node1：btype=1 (UPPER), bbound=0。执行 else: if sol[xi]>bbound。
    # 因 1.0 > 0 成立，故 d1=False（不包含最优解）。
    # 对于 Node2：btype=0 (LOWER), bbound=1。执行 if: if sol[xi]<bbound。
    # 因 1.0 < 1 不成立，故 d2=True（包含最优解）。
    # 结论：Node2 胜出，逻辑正确。
    #
    # 2. 变量名匹配验证（t_yit vs yit）
    # -------------------------------
    # 现象：分支信息中变量名为 "t_yit(18_10)"（SCIP转换后变量），
    # 而 .sol 文件中仅存储 "yit(18_10)"（原始变量名）。
    # 调试结果：
    # 尽管前缀不同，但执行 sol[bvar]（其中 bvar.name == 't_yit(...)'）时，
    # SCIP 能够自动通过内部索引映射找到对应的原始变量值。
    # 若 sol 中有值（如 1.0），则正确返回 1.0；
    # 若 sol 中无值（即 0 值变量），则正确返回默认值 0.0。
    #
    # 3. 最终结论
    # -------------------------------
    # 即使 sol 文件缺失零值变量，且变量名存在 "t_" 前缀差异，
    # 该逻辑依然能准确判断节点包含性，无需修改代码。
    # =================================================================

            
    def setOptsol(self, optsol):
        self.optsol = optsol
        

class OracleNodeSelectorEstimator_RankNet(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, device, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)

        ## 复现Abdel的原始Ranknet---------------------------------------------------------------------------------------------------------------------
        # policy = RankNet().to(device)
        # # policy.load_state_dict(torch.load(f"./learning/policy_{problem}_ranknet.pkl", map_location=device))  # run from main 源码内容
        # policy.load_state_dict(torch.load(f"D:/LiJiamigFile/Comparenodes_data/log_MLP/{problem}/policy_{problem}_ranknet.pkl", map_location=device, weights_only=True))  # 佳明修改,确保pytorch新版加载
        # print(f"load MLP: D:/LiJiamigFile/Comparenodes_data/log_MLP/{problem}/policy_{problem}_ranknet.pkl")
        # policy.eval()
        # self.policy = policy

        ## 使用注意力池化参数,让物理特征参与训练------------------------------------------------------------------------------------------------------------
        ## policy2 = RankNetNew(use_pinfo=True, n_phys=6).to(device)     ## 最终6列物理特征
        policy2 = RankNetNew(use_pinfo=True, n_phys=4).to(device)      # 最终4列物理特征
        # policy2 = RankNetNew(use_pinfo=False).to(device)                 # Nopinfo
        policy2.load_state_dict(torch.load(f"D:/LiJiamigFile/Comparenodes_data/log_MLP/{problem}/policy_{problem}_ranknet_pinfo.pkl", map_location=device, weights_only=True))  # 佳明修改,确保pytorch新版加载
        print(f"load MLP: D:/LiJiamigFile/Comparenodes_data/log_MLP/{problem}/policy_{problem}_ranknet_pinfo.pkl")
        policy2.eval()
        self.policy2 = policy2


        self.device = device
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.fea_time = 0  # 佳明新增
        self.inf_time = 0  # 佳明新增
        
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0

        # 佳明新增, 每个实例只保存1次phys(静态物理信息)
        phy_np = self.comp_featurizer.phys_features  # 共4列: pmax, pmin, pmax/Pl, pmin/Pl
        phy_np_trip = phy_np[:, 2:4]                 # 最终4列物理特征
        if phy_np.size > 0:
            self.phys_tensor = torch.tensor(
                # phy_np,      # 最终6列物理特征
                phy_np_trip,   # 最终4列物理特征
                dtype=torch.float,
                device=self.device
            )
        else:
            self.phys_tensor = None

    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1

        start_time_1 = time.time()
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))

        # 每个节点推理前提取新的变量全局信息(动态变化)
        # glob_np = self.comp_featurizer._get_var_global_info()       # 2026.3.7 版本
        # glob_trip = self.comp_featurizer._sanitize_array(glob_np)   # 剔除异常数据
        # glob_tensor = torch.tensor(glob_trip, dtype=torch.float, device=self.device)

        glob_np1 = self.comp_featurizer._get_var_local_info(node1)    # 2026.4.1 版本 Node 1 的变量边界
        glob_np2 = self.comp_featurizer._get_var_local_info(node2)    # 2026.4.1 版本 Node 2 的变量边界

        # 分别清洗，防止一个节点的异常值影响另一个
        glob_trip1 = self.comp_featurizer._sanitize_array(glob_np1)
        glob_trip2 = self.comp_featurizer._sanitize_array(glob_np2)
        # 转换为 Tensor
        glob_tensor1 = torch.tensor(glob_trip1, dtype=torch.float, device=self.device) # (N, 2)
        glob_tensor2 = torch.tensor(glob_trip2, dtype=torch.float, device=self.device) # (N, 2)

        self.fea_time += (time.time() - start_time_1)  # 佳明新增,累计特征提取的总耗时

        if self.phys_tensor is not None:

            ## 2026.3.7 版本
            # combined = torch.cat([self.phys_tensor, glob_tensor], dim=1)  # (n_vars行, n_phys列),其中n_phys=4+3; # 例如: case118的pinfo: (5184, 7)
            # t_f1 = torch.tensor(f1, dtype=torch.float, device=self.device).unsqueeze(0)  # ← 增加 batch 维, 得到: (1,20)
            # t_f2 = torch.tensor(f2, dtype=torch.float, device=self.device).unsqueeze(0)  # ← 增加 batch 维, 得到: (1,20)

            ## 2026.4.1 版本
            # 构造 Node 1 的完整输入: [Phys, N1_LB, N1_UB]
            # 形状: (N, phys_dim + 2)
            combined_node1 = torch.cat([self.phys_tensor, glob_tensor1], dim=1)
            # 构造 Node 2 的完整输入: [Phys, N2_LB, N2_UB]
            # 形状: (N, phys_dim + 2)
            combined_node2 = torch.cat([self.phys_tensor, glob_tensor2], dim=1)
            t_f1 = torch.tensor(f1, dtype=torch.float, device=self.device).unsqueeze(0)  # (1, 20)
            t_f2 = torch.tensor(f2, dtype=torch.float, device=self.device).unsqueeze(0)  # (1, 20)

            start_time_2 = time.time()

            with torch.no_grad():

                ## 2026.3.7 版本
                # decision = self.policy2(t_f1, t_f2, [combined])                          # 输入物理特征
                # decision = self.policy2(t_f1, t_f2, None)                                # 不输入物理特征

                ## 2026.4.1 版本
                # 注意：这里传入的是 [combined_node1] 和 [combined_node2]
                # 因为模型期望的是 List[Tensor]，即使 batch_size=1，也要包一层列表
                decision = self.policy2(t_f1, t_f2, [combined_node1], [combined_node2])  # 输入物理特征
                # decision = self.policy2(t_f1, t_f2, None, None)                        # 不输入物理特征

            self.inf_time += (time.time() - start_time_2)                                # 佳明新增,累计推理的总耗时

        ## 复现Abdel的原始Ranknet---------------------------------------------------------------------------------------------------------------------
        # else:
        #     start_time_2 = time.time()
        #     with torch.no_grad():
        #         decision =  self.policy(torch.tensor(f1, dtype=torch.float, device=self.device), torch.tensor(f2, dtype=torch.float, device=self.device))
        #     self.inf_time += (time.time() - start_time_2)  # 佳明新增,累计推理的总耗时
        
    
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1


