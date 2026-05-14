# utilities_LJM_Pinfo_v0 和 IEEE_g 是已有的模块
# 它们需提供：
#   - getunitdata(csv_path) → list of objects with .pmax, .pmin
#   - getbusdata(bus_csv, load_csv, sysload_csv) → (buslist, load: List[float])

import os
import csv
from typing import List, Dict, Tuple
import numpy as np

# 假设这两个模块已存在（按你提供的接口）
import utilities_LJM_Pinfo_v0
import IEEE_g


def find_milp_path(problem: str, part: str, data_dir: str):
    '''
    Args:
        problem: 例如  case118
        part:    例如  train_milp
        data_dir: 例如 f"D:\LiJiamigFile\Comparenodes_data\data\case118"

    Returns:
        list: 存放着.lp完整路径的列表
    '''
    # data_partitions = ['train_milp', 'valid_milp']  # dont change  # 佳明修改
    # data_partitions = ['valid_milp']  # dont change  # 佳明修改
    # data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case118"
    train_milp_begin = 1  # case118,  case2383 , 24GX, case1888
    train_milp_end = 2001
    valid_milp_begin = 2001
    valid_milp_end = 2401
    test_milp_begin = 2401
    test_milp_end = 2601
    # train_milp_begin = 1  # case300
    # train_milp_end = 201
    # valid_milp_begin = 201
    # valid_milp_end = 241
    # test_milp_begin = 241
    # test_milp_end = 301
    # for data_partition in data_partitions:
    instances = []
    if part == 'train_milp':
        for i in range(train_milp_begin, train_milp_end):  # 包括 1 到 xxx
            folder_name = f"{part}/{problem}_{i}"
            file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
            if os.path.isfile(file_path):
                instances.append(file_path)
    if part == 'valid_milp':
        for i in range(valid_milp_begin, valid_milp_end):  # 包括 1 到 xxx
            folder_name = f"{part}/{problem}_{i}"
            file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
            if os.path.isfile(file_path):
                instances.append(file_path)
    if part == 'test_milp':
        for i in range(test_milp_begin, test_milp_end):  # 包括 1 到 xxx
            folder_name = f"{part}/{problem}_{i}"
            file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
            if os.path.isfile(file_path):
                instances.append(file_path)
    if part == 'test_milp_Nounits':
        for i in range(test_milp_begin, test_milp_end):  # 包括 1 到 xxx
            folder_name = f"{part}/{problem}_{i}"
            file_path = os.path.join(data_dir, folder_name, f"{problem}_{i}.lp")
            if os.path.isfile(file_path):
                instances.append(file_path)
    return instances


def parse_var_dict(var_config: Dict[str, List[str]]) -> List[str]:
    """
    修改版：支持多区间生成
    例如:
        将 {'t_uit': ['1_1', '20_24', '200_1', '296_24']} 转换为变量名列表。
        逻辑：列表两两配对。
        第一对 '1_1' 到 '20_24' -> 生成 t_uit(1_1) ... t_uit(20_24)
        第二对 '200_1' 到 '296_24' -> 生成 t_uit(200_1) ... t_uit(296_24)
    """
    var_names = []

    for prefix, range_list in var_config.items():
        # 1. 检查是否为列表
        if not isinstance(range_list, list):
            raise TypeError(f"报错：Value for '{prefix}' must be a list, got {type(range_list)}")

        # 2. 检查长度是否为偶数（必须是2的倍数）
        if len(range_list) % 2 != 0:
            raise ValueError(f"报错：List for '{prefix}' must have an even number of elements (pairs). "
                             f"Got length {len(range_list)}: {range_list}")

        # 3. 两两配对处理
        # 步长为2，每次取 range_list[i] 作为起点，range_list[i+1] 作为终点
        for i in range(0, len(range_list), 2):
            start_str = range_list[i]
            end_str = range_list[i + 1]

            # 解析起点
            if '_' not in start_str:
                raise ValueError(f"Invalid start format '{start_str}' in '{prefix}'. Expected 'unit_time'")
            start_unit, start_t = map(int, start_str.split('_'))

            # 解析终点
            if '_' not in end_str:
                raise ValueError(f"Invalid end format '{end_str}' in '{prefix}'. Expected 'unit_time'")
            end_unit, end_t = map(int, end_str.split('_'))

            if start_unit > end_unit:
                raise ValueError(
                    f"配置错误: 在 '{prefix}' 中，起始机组索引 ({start_unit}) 大于 结束机组索引 ({end_unit})。\n"
                    f"请检查配置: {start_str} -> {end_str}"
                )

            if start_unit == end_unit and start_t > end_t:
                raise ValueError(
                    f"配置错误: 在 '{prefix}' 中，同一机组 ({start_unit}) 的起始时间 ({start_t}) 大于 结束时间 ({end_t})。\n"
                    f"请检查配置: {start_str} -> {end_str}"
                )

            # 生成变量
            # i 从 start_unit 到 end_unit
            for u in range(start_unit, end_unit + 1):
                # t 的范围处理：
                # 如果是起始机组，t 从 start_t 开始；否则从 1 开始
                t_start = start_t if u == start_unit else 1

                # 如果是结束机组，t 到 end_t 结束；否则到 max_time (这里假设 max_time 为 end_t 或 start_t 中的较大者，或者固定值)
                # 为了逻辑严谨，通常多区间生成时，中间完整的机组时间范围是 1~max_T。
                # 这里我们假设 end_t 是该区间的最大时间步。
                t_end = end_t if u == end_unit else max(start_t, end_t)
                # 注意：如果你的数据是固定的24小时，这里 t_end 可以直接写死或者取配置里的最大值。
                # 上面的逻辑是：如果是中间的完整机组，时间范围覆盖整个区间。

                # 更简单的逻辑：如果 start_t 和 end_t 都是 1 和 24，那中间就是 1-24。
                # 这里为了保险，使用 max(start_t, end_t) 作为完整机组的时间上限。

                for t in range(t_start, t_end + 1):
                    var_names.append(f"{prefix}({u}_{t})")

    return var_names


def generate_pinfo_csv(
        problem: str,
        lp_path: str,
        var_config: Dict[str, List[str]]
) -> None:
    """
    为单个 .lp 文件生成对应的 _Pinfo.csv
    类型提示已更新为 List[str]
    """
    if not lp_path.endswith('.lp'):
        raise ValueError(f"Input file must be .lp file, got: {lp_path}")

    if not os.path.isfile(lp_path):
        raise FileNotFoundError(f"File not found: {lp_path}")

    dir1 = os.path.dirname(lp_path)
    # basename = os.path.basename(lp_path)
    # case_name = basename[:-3]  # remove '.lp'
    case_name = problem

    # 构造所需 CSV 路径
    unit_csv = os.path.join(dir1, f"5-{case_name}-机组数据.csv")
    bus_csv = os.path.join(dir1, f"1-{case_name}-母线名称.csv")
    load_csv = os.path.join(dir1, f"2-{case_name}-母线负荷.csv")
    sysload_csv = os.path.join(dir1, f"3-{case_name}-系统负荷.csv")

    # 检查依赖文件是否存在
    for f in [unit_csv, bus_csv, load_csv, sysload_csv]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Required data file missing: {f}")

    # 加载物理数据
    try:
        unitlist = utilities_LJM_Pinfo_v0.getunitdata(unit_csv)
        buslist, load = IEEE_g.getbusdata(bus_csv, load_csv, sysload_csv)
    except Exception as e:
        raise RuntimeError(f"报错：通过 unitlist, buslist和load：Failed to load physical data for {case_name}: {e}")

    if not isinstance(load, (list, tuple)) or len(load) == 0:
        raise ValueError(f"报错：系统负荷 Load data invalid or empty for {case_name}")

    # 生成变量名列表
    try:
        var_names = parse_var_dict(var_config)
    except Exception as e:
        raise ValueError(f"报错：待提取物理信息的整数变量列表 Error parsing var_config {var_config}: {e}")

    # 准备输出 CSV 路径
    output_csv = os.path.join(dir1, f"{case_name}_Pinfo.csv")

    # 写入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(['varname', 'pmax', 'pmin', 'pmax_L', 'pmin_L'])

        for var_name in var_names:
            # 解析变量名，如 "uit(1_5)" → prefix='uit', i=1, t=5
            if '(' not in var_name or ')' not in var_name or '_' not in var_name:
                raise ValueError(f"报错：待提取物理信息的整数变量 Invalid variable name format: {var_name}")

            try:
                prefix_part, rest = var_name.split('(', 1)  # 在第一个 '(' 处将字符串切开，最多切 1 次（maxsplit=1）
                inner = rest.rstrip(')')
                i_str, t_str = inner.split('_', 1)
                i = int(i_str)
                t = int(t_str)
            except Exception as e:
                raise ValueError(f"报错：待提取物理信息的整数变量 Failed to parse variable name '{var_name}': {e}")

            # 校验索引范围
            if i < 1 or i > len(unitlist):
                raise IndexError(f"Unit index {i} out of range [1, {len(unitlist)}] for {var_name}")
            if t < 1 or t > len(load):
                raise IndexError(f"Time index {t} out of range [1, {len(load)}] for {var_name}")

            # 获取物理值
            unit = unitlist[i - 1]
            pmax = getattr(unit, 'pmax', 0.0)  # 寻找该属性,如果没有,缺省值为0.0
            pmin = getattr(unit, 'pmin', 0.0)
            load_t = getattr(load[t - 1], 'LoadSum', 0.0)

            if load_t == 0.0:
                # 避免除零
                pmax_L = float('inf') if pmax > 0 else 0.0
                pmin_L = float('inf') if pmin > 0 else 0.0
            else:
                pmax_L = pmax / load_t
                pmin_L = pmin / load_t

            def round_if_finite(x, ndigits=6):
                return round(x, ndigits) if np.isfinite(x) else x

            pmax_L = round_if_finite(pmax_L, 6)
            pmin_L = round_if_finite(pmin_L, 6)

            writer.writerow([var_name, pmax, pmin, pmax_L, pmin_L])

    print(f"Generated: {output_csv}")


def main(problem_param: str, file_path_list: List[str], var_config: Dict[str, List[str]]) -> None:
    """
    主函数：为每个 .lp 文件生成 _Pinfo.csv
    类型提示已更新
    """
    if not isinstance(file_path_list, list):
        raise TypeError("file_path_list must be a list of strings")
    if not isinstance(var_config, dict):
        raise TypeError("var_config must be a dictionary")

    if not file_path_list:
        print("Warning: file_path_list is empty. Nothing to do.")
        return

    for lp_path in file_path_list:
        if not isinstance(lp_path, str):
            raise TypeError(f"All items in file_path_list must be strings, got {type(lp_path)}")
        try:
            generate_pinfo_csv(problem_param, lp_path, var_config)
        except Exception as e:
            print(f"Failed to process {lp_path}: {e}")
            raise  # 或者 continue，根据需求决定是否中断


# ======================
# 使用示例（取消注释测试）
# ======================
if __name__ == "__main__":

    ## 注意在scip中, "t_XXX"才是默认的转换后的变量的名称!

    # case118
    problem = 'case118'
    # var_config = {
    #     't_uit': ['1_1', '54_24'],  # 修改为列表格式，单区间
    #     't_yit': ['1_1', '54_24'],
    #     't_zit': ['1_1', '54_24'],
    #     't_ycoldit': ['1_1', '54_24'],
    # }
    # partlist = ['train_milp', 'valid_milp', 'test_milp']
    var_config = {
        't_uit': ['1_1', '51_24'],  # 修改为列表格式，单区间
        't_yit': ['1_1', '51_24'],
        't_zit': ['1_1', '51_24'],
        't_ycoldit': ['1_1', '51_24'],
    }
    partlist = ['test_milp_Nounits']
    data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case118"

    # # case300
    # problem = 'case300'
    # var_config = {
    #     't_uit': ['1_1', '69_24'],  # 修改为列表格式，单区间
    #     't_yit': ['1_1', '69_24'],
    #     't_zit': ['1_1', '69_24'],
    #     't_ycoldit': ['1_1', '69_24'],
    # }
    # partlist = ['train_milp', 'valid_milp', 'test_milp']
    # # partlist = ['test_milp']
    # data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case300"

    # # case1888
    # problem = 'case1888'
    # # 示例：待提取物理信息的0-1变量名称多区间生成
    # # 区间1: 1_1 到 200_24 (保留机组)
    # # 区间2: 296_1 到 296_24 (保留机组)
    # var_config = {
    #     't_uit': ['1_1', '296_24'],
    #     't_yit': ['1_1', '148_24'],
    #     't_zit': ['1_1', '148_24'],
    #     't_ycoldit': ['1_1', '148_24'],
    # }
    #
    # # 如果想测试报错（奇数个元素）：
    # # var_config = { 't_uit': ['1_1', '20_24', '200_1'] } # 这会抛出 ValueError
    # # partlist = ['train_milp', 'valid_milp', 'test_milp']
    # partlist = ['test_milp']
    # data_dir = f"D:\LiJiamigFile\Comparenodes_data\data\case1888"


    for part in partlist:
        file_path_list = find_milp_path(problem, part, data_dir)
        print(f"{part}: {len(file_path_list)} milp")

        try:
            main(problem, file_path_list, var_config)
            print("All _Pinfo.csv files generated successfully!")
        except Exception as e:
            print(f"Program failed: {e}")