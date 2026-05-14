"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
"""

import os
import sys
import random
import numpy as np
import datetime
import pyscipopt.scip as sp
from pathlib import Path 
from functools import partial
from node_selection.node_selectors import OracleNodeSelectorAbdel
# from node_selection.recorders import LPFeatureRecorder, CompFeaturizer, CompFeaturizerSVM  # 无需GNN提取特征的函数
from node_selection.recorders import LPFeatureRecorder, CompFeaturizerSVM
from torch.multiprocessing import Process, set_start_method


def log(msg, logfile=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    if logfile is not None:
        with open(logfile, mode='a', encoding='utf-8') as f:
            f.write(line + '\n')


class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
        self.comp_behaviour_saver_svm = comp_behaviour_saver_svm
    
    # def set_LP_feature_recorder(self, LP_feature_recorder):                     # 源码内容, 佳明注释: 取消提取二部图的行为
    #     self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)

    def nodecomp(self, node1, node2):
        # comp_type有4种可能,node1=node2:0; node1>node2:-1; node1<node2:1; else,最佳估计:10
        # 调用optsol,保存真实的标签
        comp_res, comp_type = super().nodecomp(node1, node2, return_type=True)

        
        if comp_type in [-1,1]:
            # self.comp_behaviour_saver.save_comp(self.model,
            #                                     node1,
            #                                     node2,
            #                                     comp_res,
            #                                     self.counter)   # 源码内容, 佳明注释, 无需生成GNN的样本

            # 佳明注释: 调用 save_comp，内部会生成 {instance}_{comp_id}.csv 和 {instance}_Pinfo_{comp_id}.pkl
            self.comp_behaviour_saver_svm.save_comp(self.model,
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter)     # 佳明注释: 此处保存He等人的特征,已在scip.pxi的getHeHeaumeEisnerFeatures函数完成,本人原样复制过来了
        
            #print("saved comp # " + str(self.counter))
            self.counter += 1
        
        # make it bad to generate more data !
        # 佳明注释: save_comp保存的是通过optsol判断的正确标签(也可设置inv来适当扰动),
        # 如下所示,SCIP实际运行的nodecomp使用的comp_res是相反标签,达到:让SCIP探索不同路径、但每次收集的都是正确的oracle标签的目的!
        if comp_type in [-1,1]:  # 如果两个节点有其中之一被分支过,则返回node1好(-1)或者node2好(-1),然后再添加扰动
            comp_res = -1 if comp_res == 1 else 1
        else:
            comp_res = 0  # 如果两个节点从未被分支过,则返回node1=node2,无需扰动

        """
        节点比较函数官方解释https://pyscipopt.readthedocs.io/en/latest/tutorials/nodeselector.html
        compare two leaves of the current branching tree

        It should return the following values:

        value < 0, if node 1 comes before (is better than) node 2  简单来说,就是-1表示节点1更好,1表示节点2更好,0表示两者效果相等
        value = 0, if both nodes are equally good
        value > 0, if node 1 comes after (is worse than) node 2.
        """
        return comp_res


def run_episode(oracle_type, instance,  save_dir, save_dir_svm, device, logfile):
    
    model = sp.Model()
    model.hideOutput()                             # 隐藏输出
    # model.setParam("display/verblevel", 3)       # 显示的详细程度
    # model.setParam('presolving/maxrestarts', 0)  # 禁止重启
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    model.setParam('constraints/linear/upgrade/logicor',0)
    model.setParam('constraints/linear/upgrade/indicator',0)
    model.setParam('constraints/linear/upgrade/knapsack', 0)
    model.setParam('constraints/linear/upgrade/setppc', 0)
    model.setParam('constraints/linear/upgrade/xor', 0)
    model.setParam('constraints/linear/upgrade/varbound', 0)
    model.setParam('limits/time', 3600.0)
    
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))  # 创建解,但是不直接传入SCIP的当前模型 官方说明: https://pyscipopt.readthedocs.io/en/latest/api/model.html#pyscipopt.Model.readSolFile
    
    # comp_behaviour_saver = CompFeaturizer(f"{save_dir}", instance_name=str(instance).split("/")[-1])                         # 源码内容, 佳明注释, 无需生成GNN的样本
    # comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=str(instance).split("/")[-1])       # 源码内容
    instance_name = os.path.splitext(os.path.basename(instance))[0]  # 获取milp名称,例如case118_5
    # var_dict = {var.name: var for var in model.getVars(transformed=True)}  # 求解开始前, transformed=True无法生效, 因此, 请在每次提取node的特征时再临时构建,虽然慢,但是已经处于transform空间
    dir_path = os.path.dirname(instance)                   # 获取目录
    problem_tmp = instance_name.split('_')[0]              # 获取系统名称, 例如case118
    var_info_csv = os.path.join(dir_path, f"{problem_tmp}_Pinfo.csv")  # 组合得到物理信息的完整路径
    # comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=instance_name, var_info_csv=var_info_csv, var_dict=var_dict)    # 佳明修改v1
    comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=instance_name, var_info_csv=var_info_csv, var_dict=None)  # 佳明修改v2
    
    # oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm)                                                # 源码内容, 佳明注释, 无需生成GNN的样本
    oracle_ns = OracleNodeSelRecorder(oracle_type=oracle_type, comp_behaviour_saver="nothing", comp_behaviour_saver_svm=comp_behaviour_saver_svm)   # 佳明修改
    oracle_ns.setOptsol(optsol)
    # oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model, device))  # 源码内容, 佳明注释: 取消提取二部图的行为
        
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)


    # Run the optimizer
    model.optimize()
    # print(f"Got behaviour for instance  "+ str(instance).split("/")[-1] + f' with {oracle_ns.counter} comparisons' )
    log(instance_name + f': {oracle_ns.counter} comparisons, time: {model.getSolvingTime():.2f} s, node: {model.getNNodes()}', logfile)
    
    with open("nnodes.csv", "a+") as f:   # 源码内容, 佳明注释
        f.write(f"{model.getNNodes()},")
        f.close()
    with open("times.csv", "a+") as f:
        f.write(f"{model.getSolvingTime()},")
        f.close()
        
    return 1


def run_episodes(oracle_type, instances, save_dir, save_dir_svm, device, logfile):
    
    for instance in instances:
        run_episode(oracle_type, instance, save_dir, save_dir_svm, device, logfile)
        
    print("finished running episodes for process")  # 每个进程完成后独立打印1次进程结束提示,总打印次数=n_cpu
        
    return 1


def distribute(n_instance, n_cpu):
    if n_cpu == 1:
        return [(0, n_instance)]
    
    k = n_instance //( n_cpu -1 )
    r = n_instance % (n_cpu - 1 )
    res = []
    for i in range(n_cpu -1):
        res.append( ((k*i), (k*(i+1))) )
    
    res.append(((n_cpu - 1) *k ,(n_cpu - 1) *k + r ))
    return res


if __name__ == "__main__":
    

    
    oracle = 'optimal_plunger'
    problem = 'GISP'
    # data_partitions = ['train', 'valid'] #dont change  # 源码内容,
    data_partitions = ['train_milp', 'valid_milp']  # dont change  # 佳明修改
    # data_partitions = ['valid_milp']  # dont change  # 佳明修改
    data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case118"
    # data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case300"
    # data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case1888"

    # train_milp_begin = 1    # case118,  case2383 , 24GX, case1888  # 取消生成训练集
    # train_milp_end = 2001

    valid_milp_begin = 2001
    valid_milp_end = 2401
    # train_milp_begin = 1    # case300
    # train_milp_end = 201
    # valid_milp_begin = 201
    # valid_milp_end = 241
    # train_milp_begin = 1  # ljm debug
    # train_milp_end = 10
    # valid_milp_begin = 2001
    # valid_milp_end = 2005

    n_cpu = 10
    n_instance = -1
    device = 'cpu'
    
    with open("nnodes.csv", "w") as f:  # 源码内容, 佳明注释
        f.write("")
        f.close()
    with open("times.csv", "w") as f:
        f.write("")
        f.close()
        
    
    #Initializing the model 
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-oracle':
            oracle = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-data_dir':
            data_dir = str(sys.argv[i + 1])
   
    logfile = os.path.join(data_dir,f'{problem}-generate-samples.txt')
    if os.path.exists(logfile):
        print(f"Error: Log file already exists: {logfile}", file=sys.stderr)
        print("This likely means the experiment has already been run or is running.", file=sys.stderr)
        print("To avoid overwriting results, please delete the file manually or use a different output directory.",
              file=sys.stderr)
        sys.exit(1)  # 非零退出码表示错误
    log(f"MILP instance dir: {data_dir}", logfile)
  
    for data_partition in data_partitions:


        # save_dir = os.path.join(os.path.dirname(__file__), f'./data/{problem}/{data_partition}')           # 源码内容, 佳明注释
        # save_dir_svm = os.path.join(os.path.dirname(__file__), f'./data_svm/{problem}/{data_partition}')
        
        # try:                               # 源码内容, 佳明注释
        #     os.makedirs(save_dir)
        # except FileExistsError:
        #     ""
        #
        # try:
        #     os.makedirs(save_dir_svm)
        # except FileExistsError:
        #     ""
        save_dir = 'nothing'                                 # 佳明修改, 只生成MLP所需的样本, 不再生成GNN的样本
        save_dir_svm = os.path.join(data_dir, f"{data_partition}_samples")
        os.makedirs(save_dir_svm)                            # 如果旧的样本文件夹存在,则会报错,需要手动备份旧的样本文件夹再重新运行

        # n_keep  = n_instance if data_partition == 'train' or n_instance == -1 else int(0.2*n_instance)  # 源码内容, 佳明注释, 不要对train_milp或者valid_milp中的文件个数进行截断
        
        # instances = list(Path(os.path.join(os.path.dirname(__file__),
        #                                    f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))  # 源码内容
        # random.shuffle(instances)                                                                       # 源码内容, 佳明注释, 不要对train_milp或者valid_milp中的文件个数进行截断
        # instances = instances[:n_keep]

        instances = []
        # if data_partition == 'train_milp':  # 取消生成训练集
        #     for i in range(train_milp_begin, train_milp_end):  # 包括 1 到 xxx
        #         folder_name = f"{data_partition}/{problem}_{i}"
        #         file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
        #         if os.path.isfile(file_path):
        #             instances.append(file_path)
        if data_partition == 'valid_milp':
            for i in range(valid_milp_begin, valid_milp_end):  # 包括 1 到 xxx
                folder_name = f"{data_partition}/{problem}_{i}"
                file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
                if os.path.isfile(file_path):
                    instances.append(file_path)


        # print(f"Generating {data_partition.upper()} samples from {len(instances)} instances using oracle {oracle}")
        log(f"Generating {data_partition.upper()} samples from {len(instances)} instances using oracle {oracle}", logfile)

        processes = [  Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                        oracle_type=oracle,
                                                        instances=instances[ p1 : p2], 
                                                        save_dir=save_dir,
                                                        save_dir_svm=save_dir_svm,
                                                        device=device,
                                                        logfile=logfile))
                        for p,(p1,p2) in enumerate(distribute(len(instances), n_cpu))]
        
        
        try:
            set_start_method('spawn')
        except RuntimeError:
            ''
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
            
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    # print(f"Mean number of node created  {np.mean(nnodes)}")
    # print(f"Mean solving time  {np.mean(times)}")
    # print(f"Median number of node created  {np.median(nnodes)}")
    # print(f"Median solving time  {np.median(times)}")
    log(f"Mean number of node created  {np.mean(nnodes)}", logfile)
    log(f"Mean solving time  {np.mean(times)}", logfile)
    log(f"Median number of node created  {np.median(nnodes)}", logfile)
    log(f"Median solving time  {np.median(times)}", logfile)
    
    
                         
            
        

        
