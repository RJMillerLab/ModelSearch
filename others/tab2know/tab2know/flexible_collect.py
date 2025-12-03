import os, sys, csv, json
try:
    from .collect import get_file_meta
except ImportError:
    from collect import get_file_meta

BASE = "http://karmaresearch.net/"

def find_csv_directory(basedir):
    """
    智能查找CSV目录，支持多种结构：
    1. basedir/csv/ (原始结构)
    2. basedir/ (直接包含CSV文件)
    3. basedir/csvs/ (你的需求)
    4. basedir/tables/ (其他可能的结构)
    """
    possible_dirs = [
        os.path.join(basedir, 'csv'),      # 原始结构
        basedir,                           # 直接包含CSV
        os.path.join(basedir, 'csvs'),     # 你的需求
        os.path.join(basedir, 'tables'),   # 其他可能
        os.path.join(basedir, 'data'),     # 其他可能
    ]
    
    for csv_dir in possible_dirs:
        if os.path.exists(csv_dir) and os.path.isdir(csv_dir):
            # 检查是否包含CSV文件
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                # print(f"找到CSV目录: {csv_dir} (包含 {len(csv_files)} 个CSV文件)")
                return csv_dir
    
    # 如果都没找到，返回原始结构（保持向后兼容）
    print(f"警告: 未找到CSV文件，使用默认路径: {os.path.join(basedir, 'csv')}")
    return os.path.join(basedir, 'csv')

def get_all_metadata_flexible(basedir, paper_prefix=''):
    """
    灵活版本的get_all_metadata，支持多种目录结构
    """
    # 智能查找CSV目录
    csv_dir = find_csv_directory(basedir)
    
    # 读取ID映射文件（如果存在）
    f = os.path.join(basedir, 'id_lookup.txt')
    if not os.path.isfile(f):
        id_lookup = {}
    else:
        id_lookup = {
            a: b
            for line in open(f) for a, b in [line.strip().split()]
        }

    # 处理CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"警告: 在 {csv_dir} 中未找到CSV文件")
        return
    
    for name in csv_files:
        fname = os.path.join(csv_dir, name)
        if os.path.isfile(fname):
            yield get_file_meta(fname,
                                paper_prefix=paper_prefix,
                                id_lookup=id_lookup)

def get_csv_count_flexible(basedir):
    """
    灵活版本的CSV文件计数
    """
    csv_dir = find_csv_directory(basedir)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    return len(csv_files)
