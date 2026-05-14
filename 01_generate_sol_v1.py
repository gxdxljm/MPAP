import os
import sys
import argparse
import pathlib
from collections import defaultdict
import gzip
import pickle
import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import openpyxl

# Try importing solvers
try:
    from pyscipopt import Model as SCIPModel
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def log(msg, logfile=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    if logfile is not None:
        with open(logfile, mode='a', encoding='utf-8') as f:
            f.write(line + '\n')



# ==============================
# SCIP Solver Wrapper
# ==============================
def solve_with_scip(lp_file: str, set_file: str):
    lp_path = pathlib.Path(lp_file)
    lp_name = os.path.basename(lp_file)
    if not lp_path.exists():
        return {
            'lp_file': str(lp_path),
            'solver': 'SCIP',
            'status': 'FileNotFound',
            'time': None,
            'gap': None,
            'nodes': None,
            'sol_written': False,
            'lp_name': lp_name
        }

    try:
        model = SCIPModel()
        # model.hideOutput()  # 是否隐藏输出日志

        if set_file != "nothing" and os.path.isfile(set_file):
            model.readParams(set_file)

        model.readProblem(str(lp_path))
        model.optimize()
        solving_time = model.getSolvingTime()

        status = model.getStatus()
        best_solution = model.getBestSol()
        sol_written = False
        if best_solution is not None:
            sol_path = lp_path.with_suffix('.sol')
            model.writeBestSol(str(sol_path))  # pyscipopt接口,调用的是c函数SCIPprintBestSol,网址: https://scipopt.org/doc/html/scip__sol_8c_source.php
            sol_written = True

        gap = model.getGap() * 100

        nodes = model.getNNodes()

        return {
            'lp_file': str(lp_path),
            'solver': 'SCIP',
            'status': str(status),
            'time': solving_time,
            'gap': gap,
            'nodes': nodes,
            'sol_written': sol_written,
            'lp_name': lp_name
        }

    except Exception as e:
        return {
            'lp_file': str(lp_path),
            'solver': 'SCIP',
            'status': f'Error: {str(e)}',
            'time': None,
            'gap': None,
            'nodes': None,
            'sol_written': False,
            'lp_name': lp_name
        }


# ==============================
# Gurobi Solver Wrapper
# ==============================
def solve_with_gurobi(lp_file: str):
    lp_path = pathlib.Path(lp_file)
    lp_name = os.path.basename(lp_file)
    if not lp_path.exists():
        return {
            'lp_file': str(lp_path),
            'solver': 'Gurobi',
            'status': 'FileNotFound',
            'time': None,
            'gap': None,
            'nodes': None,
            'sol_written': False,
            'lp_name': lp_name
        }

    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model(env=env)
        model.read(str(lp_path))

        # Set default parameters
        model.Params.MIPGap = 0.0
        model.Params.TimeLimit = 1000.0

        model.optimize()

        solving_time = model.Runtime
        sol_written = False
        if model.SolCount > 0:
            sol_path = lp_path.with_suffix('.sol')
            model.write(str(sol_path))
            sol_written = True

        gap = model.MIPGap * 100

        nodes = model.NodeCount if hasattr(model, 'NodeCount') else 0

        status_map = {
            GRB.OPTIMAL: 'OPTIMAL',
            GRB.TIME_LIMIT: 'TIME_LIMIT',
            GRB.INFEASIBLE: 'INFEASIBLE',
            GRB.UNBOUNDED: 'UNBOUNDED',
            GRB.INTERRUPTED: 'INTERRUPTED'
        }
        status_str = status_map.get(model.Status, str(model.Status))

        return {
            'lp_file': str(lp_path),
            'solver': 'Gurobi',
            'status': status_str,
            'time': solving_time,
            'gap': gap,
            'nodes': nodes,
            'sol_written': sol_written,
            'lp_name': lp_name
        }

    except Exception as e:
        return {
            'lp_file': str(lp_path),
            'solver': 'Gurobi',
            'status': f'Error: {str(e)}',
            'time': None,
            'gap': None,
            'nodes': None,
            'sol_written': False,
            'lp_name': lp_name
        }
    finally:
        if 'env' in locals():
            env.dispose()


# ==============================
# Delete existing .sol files
# ==============================
def delete_existing_sol_files(lp_files):
    sol_files = []
    for lp in lp_files:
        sol = pathlib.Path(lp).with_suffix('.sol')
        if sol.exists():
            sol_files.append(sol)

    print(f"\nFound {len(sol_files)} existing .sol files. Deleting...")
    for sol in sol_files:
        try:
            os.remove(sol)
            print(f"Deleted: {sol}")
        except Exception as e:
            print(f"Failed to delete {sol}: {e}")
    print("Deletion completed.\n")


# ==============================
# Main function
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Solve .lp files with SCIP or Gurobi.")
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--folder_name', type=str, default="train_MC_UE")
    parser.add_argument('--second_filer_pct', type=float, default=1.0)
    parser.add_argument('--set_file', type=str, default="nothing", help="Only used if solver=scip")
    parser.add_argument('--solver', type=str, choices=['scip', 'gurobi'], default='scip',
                        help="Choose solver: 'scip' or 'gurobi'")
    parser.add_argument('--delete_sol', type=lambda x: x.lower() == 'true', default=False,
                        help="Delete existing .sol files before solving (true/false)")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of parallel processes")

    args = parser.parse_args()

    if args.solver == 'scip' and not SCIP_AVAILABLE:
        raise RuntimeError("SCIP (pyscipopt) is not available. Please install it or use --solver gurobi.")
    if args.solver == 'gurobi' and not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi is not available. Please install it or use --solver scip.")


    # train_milp_begin = 1    # case118,  case2383 , 24GX, case1888
    # train_milp_end = 2001
    # valid_milp_begin = 2001
    # valid_milp_end = 2401
    # test_milp_begin = 2401
    # test_milp_end = 2601

    # train_milp_begin = 1    # 取消train和valid集的oracle生成
    # train_milp_end = 2001
    # valid_milp_begin = 2001
    # valid_milp_end = 2401
    test_milp_begin = 2470
    test_milp_end = 2485

    # train_milp_begin = 1    # case300
    # train_milp_end = 201
    # valid_milp_begin = 201
    # valid_milp_end = 241
    # test_milp_begin = 241
    # test_milp_end = 301

    # train_milp_begin = 201    # case118,  case2383 , 24GX, case1888
    # train_milp_end = 401
    # valid_milp_begin = 2041
    # valid_milp_end = 2081
    # test_milp_begin = 2421
    # test_milp_end = 2451

    # 样本保存路径,日志文件路径
    out_dir = args.data_dir  # 不允许将data生成在.py文件所在的文件夹！
    logfile = os.path.join(out_dir,f'{args.problem}-generate-sol.txt')
    if os.path.exists(logfile):
        print(f"Error: Log file already exists: {logfile}", file=sys.stderr)
        print("This likely means the experiment has already been run or is running.", file=sys.stderr)
        print("To avoid overwriting results, please delete the file manually or use a different output directory.",
              file=sys.stderr)
        sys.exit(1)  # 非零退出码表示错误
    log(f"MILP instance dir: {args.data_dir}", logfile)

    # 一旦无法完整生成.sol,则重新运行该.py文件
    instances_all = []


    # for i in range(train_milp_begin, train_milp_end):  # 包括 1 到 xxx  # 取消train和valid集的oracle生成
    #     folder_name = f"train_milp/{args.problem}_{i}"
    #     file_path = os.path.join(args.data_dir, folder_name, f"{args.problem}_{i}.lp")
    #     if os.path.isfile(file_path):
    #         instances_all.append(file_path)
    # for i in range(valid_milp_begin, valid_milp_end):  # 包括 xxx+1 到 yyy
    #     folder_name = f"valid_milp/{args.problem}_{i}"
    #     file_path = os.path.join(args.data_dir, folder_name, f"{args.problem}_{i}.lp")
    #     if os.path.isfile(file_path):
    #         instances_all.append(file_path)


    for i in range(test_milp_begin, test_milp_end):  # 包括 yyy+1 到 zzz
        folder_name = f"test_milp/{args.problem}_{i}"
        file_path = os.path.join(args.data_dir, folder_name, f"{args.problem}_{i}.lp")
        if os.path.isfile(file_path):
            instances_all.append(file_path)

    log(f"MILP total num: {len(instances_all)}", logfile)

    # Step 3: Optionally delete existing .sol
    if args.delete_sol:
        delete_existing_sol_files(instances_all)

    # Step 4: Solve in parallel
    log(f"Starting solving with {args.n_jobs} parallel jobs using {args.solver.upper()}...\n", logfile)  # .upper()用于把str字符串中的小写字母全部转化为大写字母

    results = []
    if args.n_jobs == 1:
        # Sequential
        for lp in instances_all:
            if args.solver == 'scip':
                res = solve_with_scip(lp, args.set_file)
            else:
                res = solve_with_gurobi(lp)
            results.append(res)
            msg = f"[{res['solver']}] {os.path.basename(lp)} → Status: {res['status']}"
            if res['time'] is not None:
                msg += f" | Time: {res['time']:.2f}s | Gap: {res['gap']:.4f}% | Nodes: {res['nodes']}"
            print(msg)  # 每个算例的结果
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            if args.solver == 'scip':
                futures = {executor.submit(solve_with_scip, lp, args.set_file): lp for lp in instances_all}
            else:
                futures = {executor.submit(solve_with_gurobi, lp): lp for lp in instances_all}

            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                msg = f"[{res['solver']}] {os.path.basename(res['lp_file'])} → Status: {res['status']}"
                if res['time'] is not None:
                    msg += f" | Time: {res['time']:.2f}s | Gap: {res['gap']:.4f}% | Nodes: {res['nodes']}"
                print(msg)  # 每个算例的结果

    # Step 5: Summary statistics
    valid_results = [r for r in results if r['time'] is not None and r['gap'] is not None]
    total = len(results)
    solved = len(valid_results)

    if solved > 0:
        avg_time = sum(r['time'] for r in valid_results) / solved
        avg_gap = sum(r['gap'] for r in valid_results) / solved
        avg_nodes = sum(r['nodes'] for r in valid_results) / solved
        non_optimal = sum(1 for r in valid_results if r['gap'] > 1e-4)
        non_optimal_milp = [r['lp_name'] for r in valid_results if r['gap'] > 1e-4]
    else:
        avg_time = avg_gap = avg_nodes = float('nan')
        non_optimal = 0
        non_optimal_milp = ['It is wrong...']

    log("\n" + "="*60, logfile)
    log("SUMMARY", logfile)
    log("="*60, logfile)
    log(f"Solver               : {args.solver.upper()}", logfile)
    log(f"set_file_path:       : {args.set_file}", logfile)
    log(f"Total instances      : {len(instances_all)}", logfile)
    log(f"Total solution file  : {total}", logfile)
    log(f"Solved successfully  : {solved}", logfile)
    log(f"Average time (s)     : {avg_time:.2f}", logfile)
    log(f"Average gap (%)      : {avg_gap:.4f}", logfile)
    log(f"Average nodes        : {avg_nodes:.0f}", logfile)
    log(f"Non-optimal (gap>0.01percent)  : {non_optimal}", logfile)
    # log(f"Non-optimal-milp (gap>0.01percent)  : {non_optimal_milp}", logfile)
    log("="*60)

    # Step 6: Save to Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(out_dir, f"{args.problem}-generate-sol.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Instance Results', index=False)
        summary_df = pd.DataFrame({
            'Metric': [
                'Solver',
                'Total Instances',
                'Solved Successfully',
                'Average Time (s)',
                'Average Gap (%)',
                'Average Nodes',
                'Non-optimal Instances (gap > 1e-4%)'
            ],
            'Value': [
                args.solver.upper(),
                total,
                solved,
                avg_time,
                avg_gap,
                avg_nodes,
                non_optimal
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    log(f"Summary results saved to: {excel_path}", logfile)


if __name__ == '__main__':
    main()

## 使用示例1:
# 在readme.md中使用 SCIP + 删除旧解 + 4 进程
# python 01_generate_sol_v1.py \
#   --problem case118 \
#   --set_file "C:\Users\LiJiangMing\Desktop\nodeselect.set"
#   --data_dir ./data \
#   --solver scip \
#   --set_file ./configs/default.set \
#   --delete_sol true \
#   --n_jobs 4

## 使用示例2:
# 在readme.md中使用 Gurobi + 不删解 + 单进程
# python 01_generate_sol_v1.py \
#   --problem case118 \
#   --set_file "C:\Users\LiJiangMing\Desktop\nodeselect.set"
#   --data_dir ./data \
#   --solver gurobi \
#   --delete_sol false \
#   --n_jobs 1