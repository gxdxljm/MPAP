import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def pkl_to_csv_visualization(pkl_folder: str, output_base_dir: str = None,
                             var_count: int = None, phys_feature_count: int = 6):
    """
    将指定文件夹下的所有 Pinfo_{comp_id}.pkl 文件转换为可视化 CSV。

    Args:
        pkl_folder (str): 包含 .pkl 文件的文件夹路径。
        output_base_dir (str): 输出 CSV 的根目录。如果为 None，则输出到原文件夹的同级。
        var_count (int): 变量总数（用于生成行索引名称）。如果为 None，尝试从文件名或数据推断。
        phys_feature_count (int): 物理特征的列数（默认 6）。根据你的实际物理特征列数调整。
    """
    pkl_path = Path(pkl_folder)
    if not pkl_path.exists():
        print(f"错误：路径 {pkl_folder} 不存在。")
        return

    # 如果未指定输出目录，默认在原目录下创建 visual_csv 文件夹
    if output_base_dir is None:
        output_root = pkl_path.parent / f"{pkl_path.stem}_visual_csv"
    else:
        output_root = Path(output_base_dir)
    output_root.mkdir(exist_ok=True)

    # 查找所有符合你命名规则的文件
    # 假设文件名为: <instance_name>_Pinfo_<comp_id>.pkl
    pkl_files = list(pkl_path.glob("*_Pinfo_*.pkl"))

    if not pkl_files:
        print(f"在 {pkl_folder} 中未找到 *_Pinfo_*.pkl 文件。")
        return

    print(f"[*] 找到 {len(pkl_files)} 个文件，开始转换...")

    # 定义边界列的名称（对应你代码中的 lb_1, ub_1, lb_2, ub_2）
    boundary_columns = ['Node1_lb', 'Node1_ub', 'Node2_lb', 'Node2_ub']

    for pkl_file in pkl_files:
        try:
            # 1. 读取 .pkl 文件 (NumPy array)
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # 确保 data 是 2D 数组
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_vars, n_features = data.shape

            # 2. 构建列名
            # 推断物理特征列名
            # if var_count is not None and phys_feature_count > 0:
            #     # 如果你知道物理特征列数，生成 Var1_F1, Var1_F2... 这样的列名
            #     # 或者你可以直接写死物理特征名，例如：['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
            #     # 这里为了通用性，使用通用命名
            #     physical_columns = [f'Phys_Feature_{i + 1}' for i in range(phys_feature_count)]
            # else:
            #     # 如果没有物理特征，或者维度不匹配，只保留边界列
            #     phys_feature_count = 0
            #     physical_columns = []
            ## 佳明修复
            if phys_feature_count > 0:
                physical_columns = [f'Phys_Feature_{i + 1}' for i in range(phys_feature_count)]
            else:
                physical_columns = []


            # 最终列名 = 物理特征列 + 边界列
            # 逻辑: [Phys_F1...Phys_Fn, Node1_lb, Node1_ub, Node2_lb, Node2_ub]
            column_names = physical_columns + boundary_columns

            # 3. 构建 DataFrame
            # 注意：你的数据结构是 (n_variables, n_features)
            # 我们假设 n_features = phys_feature_count + 4 (4个边界)
            df = pd.DataFrame(data, columns=column_names)

            # 如果有变量名信息，可以作为索引（行名）
            # 这里使用 Var_0, Var_1... 作为行索引
            df.index.name = 'Variable_Index'
            df.insert(0, 'Variable_Name', [f'Var_{i}' for i in range(n_vars)])

            # 4. 生成输出路径
            # 输出文件名：原文件名.csv
            csv_filename = pkl_file.stem + '.csv'
            output_dir = output_root
            output_path = output_dir / csv_filename

            # 5. 保存为 CSV
            df.to_csv(output_path, index=False)
            print(f"[OK] 已保存: {output_path} (Shape: {data.shape})")

        except Exception as e:
            print(f"[ERROR] 处理 {pkl_file} 时出错: {e}")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 1. 修改为你的 .pkl 文件所在的文件夹路径
    #    也就是 CompFeaturizerSVM.save_dir 所指向的目录
    # PKL_SOURCE_DIR = r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_data\samples\your_instance_name"
    PKL_SOURCE_DIR = r"D:\LiJiamigFile\Comparenodes_data\data_pkl2csv\case118_pkl"

    # 2. 输出目录（可选）。设为 None 则自动在源目录同级创建文件夹
    # OUTPUT_DIR = None
    OUTPUT_DIR = r"D:\LiJiamigFile\Comparenodes_data\data_pkl2csv\case118_csv"


    # 3. 关键参数：你的物理特征有多少列？
    #    请根据你传入的 var_info_csv 中，除去第一列（变量名）后的列数填写。
    #    例如：如果 CSV 里有 Name, F1, F2, F3, F4, F5, F6 -> 这里填 6
    #    如果没有物理特征，填 0
    PHYSICAL_FEATURE_COLUMNS = 4

    # --- 执行 ---
    pkl_to_csv_visualization(
        pkl_folder=PKL_SOURCE_DIR,
        output_base_dir=OUTPUT_DIR,
        phys_feature_count=PHYSICAL_FEATURE_COLUMNS
    )