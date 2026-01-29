# single_task.py
import sys
import os
import gc
import time
import numpy as np
import torch
import distance_transforms as dts

from pathlib import Path

def get_image_paths(image_dir="./image/", extensions=None, recursive=False):
    ext = {'.png'} if extensions is None else extensions
    path = Path(image_dir)
    files = path.rglob("*") if recursive else path.iterdir()
    result = [str(f) for f in files if f.is_file() and f.suffix.lower() in ext and not f.name.startswith('.')]
    return sorted(result) if result else []

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_binary_matrix(size, black_percent, seed=42):
    """生成二值化矩阵（兼容占比0%场景）"""
    black_ratio = black_percent / 100.0
    if black_ratio <= 0:
        return np.zeros((size, size), dtype=np.uint8)
    
    rng = np.random.default_rng(seed)

    random_matrix = rng.rand(size, size)
    binary_matrix = (random_matrix < black_ratio).astype(np.uint8)
    return binary_matrix

def calculate_required_memory(size, dtype=torch.uint8):
    """计算张量所需显存（MiB），针对8192尺寸预检"""
    if isinstance(size, int):
        element_count = size * size
    else:
        element_count = np.prod(size)
    
    if dtype == torch.uint8:
        bytes_per_element = 1
    else:
        bytes_per_element = 1  
    
    total_mb = (element_count * bytes_per_element) / (1024 * 1024)
    return total_mb

def get_available_cuda_memory():
    """获取可用CUDA显存（MiB）"""
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        used_mem = torch.cuda.memory_allocated(0)
        available_mem = (total_mem - used_mem) / (1024 * 1024)
        return available_mem
    except Exception:
        return 0.0

def cal_edm_single(matrix):
    """单个矩阵的EDM计时（优化大尺寸+全0矩阵兼容性）"""
    if not torch.cuda.is_available():
        return 0.0

    with torch.no_grad():
        try:
            tensor = torch.tensor(matrix, device='cuda', dtype=torch.uint8)
            torch.cuda.synchronize()  
            start = time.time()
            dts.transform_cuda(tensor)
            torch.cuda.synchronize()
            one_time = time.time() - start
        except torch.OutOfMemoryError:
            return -1  
        except Exception:
            return 0.0

    del tensor
    torch.cuda.empty_cache()
    gc.collect()
    return one_time

def get_avg_time(n, size, black_percent):
    """计算平均耗时（针对8192尺寸+0%占比增加容错）"""
    if size == 8192:
        required_mb = calculate_required_memory(size)
        available_mb = get_available_cuda_memory()
        if required_mb > (available_mb - 100):
            return -1  
    
    sum_time = 0.0
    fail_count = 0
    for _ in range(n):
        matrix = generate_binary_matrix(size, black_percent)
        one_time = cal_edm_single(matrix)
        
        if one_time == -1:
            fail_count += 1
            if fail_count >= 3: 
                return -1
            continue  
        sum_time += one_time

    if sum_time == 0.0 and fail_count == n:
        return 0.0
    valid_count = n - fail_count
    if valid_count <= 0:
        return 0.0
    avg_time_ms = (sum_time / valid_count) * 1000
    return avg_time_ms

if __name__ == "__main__":
    # 接收命令行参数：size, percent, n
    if len(sys.argv) != 4:
        print("0.0")
        sys.exit(1)

    try:
        size = int(sys.argv[1])
        percent = int(sys.argv[2])
        n = int(sys.argv[3])

        if percent < 0:
            percent = 0
        if percent > 100:
            percent = 100
        avg_time_ms = get_avg_time(n, size, percent)
        
        if avg_time_ms == -1:
            print("OOM")  
        else:
            print(f"{avg_time_ms:.2f}")
    except Exception as e:
        print("0.0")
        sys.exit(1)
