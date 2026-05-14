"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2comparenodes).
Modified for MPAP (Multi-layer Perceptron with Attention Pooling) node selection framework under the same MIT License.
"""


import sys
import os
import re
import numpy as np
import datetime
import torch
from torch.multiprocessing import Process, set_start_method
from functools import partial
from utils import record_stats, display_stats, distribute
from pathlib import Path 


def log(msg, logfile=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    if logfile is not None:
        with open(logfile, mode='a', encoding='utf-8') as f:
            f.write(line + '\n')


if __name__ == "__main__":
    
    n_cpu = 16
    n_instance = -1
    # nodesels = ['ranknet_dummy_nprimal=2']
    nodesels = []
    
    problem = 'GISP'
    normalize = True
    
    # data_partition = 'transfer'
    data_partition = 'test_milp'
    # test_milp_begin = 2401    # case118,  case2383 , 24GX, case1888
    # test_milp_end = 2601
    test_milp_begin = 1      # case300
    test_milp_end = 2601
    data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case118"
    save_dir = f'D:\LiJiamigFile\Comparenodes_data\log_test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose = False
    on_log = False
    default = False
    # delete = False ## 源码内容

    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-nodesels':
            nodesels = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        # if sys.argv[i] == '-data_partition':
        #     data_partition = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-verbose':
            verbose = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-on_log':
            on_log = bool(int(sys.argv[i + 1]))    
        if sys.argv[i] == '-default':
            default = bool(int(sys.argv[i + 1]))  
        # if sys.argv[i] == '-delete': ## 源码内容
        #     delete = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-save_dir':
            save_dir = str(sys.argv[i + 1])
        if sys.argv[i] == '-data_dir':
            data_dir = str(sys.argv[i + 1])
        if sys.argv[i] == '-test_milp_begin':
            test_milp_begin = str(sys.argv[i + 1])
        if sys.argv[i] == '-test_milp_end':
            test_milp_end = str(sys.argv[i + 1])

    ## 源码内容
    # if delete:
    #     try:
    #         import shutil
    #         # shutil.rmtree(os.path.join(os.path.abspath(''),
    #         #                            f'stats/{problem}'))    # 递归删除stats/GISP/下的所有内容
    #         shutil.rmtree(os.path.join(save_dir))  # 递归删除stats/GISP/下的所有内容
    #     except:
    #         ''



    # instances = list(Path(os.path.join(os.path.abspath(''),
    #                                    f"./problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))  # 源码内容
    instances = []
    for i in range(test_milp_begin, test_milp_end):  # 包括 1 到 xxx
        folder_name = f"{data_partition}/{problem}_{i}"
        file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
        if os.path.isfile(file_path):
            instances.append(file_path)


    if n_instance == -1 :
        n_instance = len(instances)

    import random
    random.shuffle(instances)
    instances = instances[:n_instance]

    # 测试日志
    logfile = os.path.join(save_dir,f'{problem}-test-results.txt')
    if os.path.exists(logfile):
        print(f"Error: Log file already exists: {logfile}", file=sys.stderr)
        print("This likely means the experiment has already been run or is running.", file=sys.stderr)
        print("To avoid overwriting results, please delete the file manually or use a different output directory.",
              file=sys.stderr)
        sys.exit(1)  # 非零退出码表示错误
    log("Evaluation", logfile)
    log(f"  MILP instance dir:          {data_dir}", logfile)
    log(f"  Problem:                    {problem}", logfile)
    log(f"  n_instance/problem:         {len(instances)}", logfile)
    log(f"  Nodeselectors evaluated:    {','.join( ['default' if default else '' ] + nodesels)}", logfile)
    log(f"  Device for GNN inference:   {device}", logfile)
    log(f"  Normalize features:         {normalize}", logfile)
    log("----------------", logfile)



    # In[92]:


    #Run benchmarks

    processes = [  Process(name=f"worker {p}", 
                           target=partial(record_stats,
                                          nodesels=nodesels,
                                          instances=instances[p1:p2], 
                                          problem=problem,
                                          device=torch.device(device),
                                          normalize=normalize,
                                          verbose=verbose,
                                          save_dir=save_dir,
                                          default=default))
                    for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]  


    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     ''
    #
    # a = list(map(lambda p: p.start(), processes)) #run processes
    # b = list(map(lambda p: p.join(), processes)) #join processes
    # 设置启动方法（仅需一次）
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # 已设置过
    # 启动所有进程
    for p in processes:
        p.start()
    # 等待所有进程结束
    for p in processes:
        p.join()


    # min_n = min([ int( str(instance).split('=')[1 if problem == "GISP" else 2 ].split('_')[0] )  for instance in instances ] )  # 源码内容
    #
    # max_n = max([ int( str(instance).split('=')[1 if problem == "GISP" else 2].split('_')[0] )  for instance in instances ] )
    #
    # display_stats(problem, nodesels, instances, min_n, max_n, default=default)                              # 源码内容
    # display_stats(problem, nodesels, instances, test_milp_begin, test_milp_end, save_dir, default=default)  # 打印汇总的测试结果(要启用源码record_stats_instance函数自带的np.savetxt)
    print("All tests have been successfully completed!")
   
