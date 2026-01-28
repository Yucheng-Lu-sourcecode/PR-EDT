import torch
import time
import pandas as pd
import numpy as np
import distance_transforms as dts

def read_image_to_binary_matrix(image_path):
    # 延迟导入cv2（方便确认安装成功）
    try:
        import cv2
    except ImportError:
        raise ImportError("未检测到cv2模块，请先执行 pip install opencv-python 安装")
    
    # 1. 读取灰度图（黑白图片灰度图只有0和255两个值）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}，请检查路径是否正确")
    
    # 2. 黑白图片精准转0/1二值矩阵（纯黑→1，纯白→0）
    binary_matrix = np.where(img == 0, 1, 0).astype(np.uint8)
    
    # 3. 验证转换结果（可选，便于调试）
    unique_vals = np.unique(binary_matrix)
    if not set(unique_vals).issubset({0, 1}):
        print(f"警告：图片 {image_path} 非纯黑白，已自动二值化处理")
        # 兼容非纯黑白图片，补充阈值二值化
        _, binary_matrix = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        binary_matrix = binary_matrix.astype(np.uint8)
    
    return binary_matrix

# 计算EDM并统计：耗时 + 包含所有分配的CUDA峰值内存（核心优化：扩大统计边界）
def cal_edm(matrix):
    one_time = 0.0
    memory_allocated_bytes = 0.0
    memory_reserved_bytes = 0.0
    memory_peek_allocated_bytes = 0.0  # 包含所有CUDA分配的峰值内存
    
    if torch.cuda.is_available():
        # 1. 清空缓存，保证测试独立性（清理上一次运行的残留内存）
        torch.cuda.empty_cache()
        
        # 2. 重置CUDA峰值内存计数器（关键：清零历史峰值，确保本次统计无干扰）
        # 该函数会重置当前设备的max_memory_allocated记录，从0开始统计
        torch.cuda.reset_max_memory_allocated(device=torch.cuda.current_device())
        
        # 3. 【统计边界前移】：从张量创建前就开始统计，确保捕获所有分配操作
        start = time.time()
        
        # 4. 转换为CUDA张量（这部分内存会被计入峰值）
        tensor = torch.tensor(matrix, device='cuda')
        
        # 5. 执行EDM（内部所有内存分配都会被计入峰值）
        GPU_result = dts.transform_cuda(tensor)
        
        # 6. 结束计时
        end = time.time()
        
        # 7. 计算耗时
        one_time = end - start
        
        # 8. 监控GPU内存（重点：获取包含所有分配的峰值内存）
        device = torch.cuda.current_device()
        # 当前已分配内存（实时值）
        memory_allocated_bytes = torch.cuda.memory_allocated(device)
        # 当前预留内存（GPU缓存池大小）
        memory_reserved_bytes = torch.cuda.memory_reserved(device)
        # 核心：获取本次运行中【所有CUDA分配】的内存峰值（张量+EDM内部+中间结果）
        memory_peek_allocated_bytes = torch.cuda.max_memory_allocated(device)
        
        # 9. 释放张量，清理缓存（不影响已记录的峰值）
        del GPU_result
        del tensor
        torch.cuda.empty_cache()
    
    return one_time, memory_allocated_bytes, memory_reserved_bytes, memory_peek_allocated_bytes

# 单张图片多次测试，获取平均耗时 + 最大峰值内存（包含所有分配）
def get_avg_time_and_memory(n, image_path):
    print(f"  正在读取图片：{image_path.split('/')[-1]}")
    binary_matrix = read_image_to_binary_matrix(image_path)
    size = binary_matrix.shape[0]
    print(f"  图片尺寸：{size}x{size}")
    
    sum_time = 0.0
    sum_allocated = 0.0
    sum_reserved = 0.0
    max_peek_allocated = 0.0  # 多次测试中的最大峰值（最能反映真实内存需求）
    
    for i in range(n):
        print(f"  第 {i+1}/{n} 次测试...", end="\r")
        one_time, allocated, reserved, peek_allocated = cal_edm(binary_matrix)
        
        sum_time += one_time
        sum_allocated += allocated
        sum_reserved += reserved
        
        # 更新最大峰值（确保捕获所有场景下的最大内存消耗）
        if peek_allocated > max_peek_allocated:
            max_peek_allocated = peek_allocated
    
    print()  # 清理换行
    
    # 单位转换（MB，1MB=1024^2字节）
    avg_time = sum_time / n
    avg_allocated_mb = sum_allocated / n / (1024 ** 2)
    avg_reserved_mb = sum_reserved / n / (1024 ** 2)
    peek_allocated_mb = max_peek_allocated / (1024 ** 2)
    
    return avg_time, avg_allocated_mb, avg_reserved_mb, peek_allocated_mb

if __name__ == "__main__":
    # 配置参数
    image_dir = "./image/"
    image_files = ["L-1024.png", "L-2048.png", "L-4096.png", "L-8192.png"]
    n = 10  # 每张图片测试次数
    image_paths = [f"{image_dir}{fname}" for fname in image_files]
    
    # 初始化DataFrame（明确标注峰值内存包含所有分配）
    columns = ['平均耗时(ms)', '平均已分配内存(MB)', '平均预留内存(MB)', 'CUDA峰值内存(含所有分配)(MB)']
    df = pd.DataFrame(index=image_files, columns=columns)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告：未检测到可用的GPU，无法获取CUDA内存信息！")
    else:
        print(f"检测到可用GPU：{torch.cuda.get_device_name(0)}")
        print("="*60)
        print("注意：CUDA峰值内存统计包含【张量创建+EDM内部+所有中间结果】的内存分配")
        print("="*60)
    
    # 遍历处理所有图片
    for img_path, img_name in zip(image_paths, image_files):
        print(f"正在处理图片：{img_name}")
        print("-"*50)
        try:
            avg_time, avg_allocated_mb, avg_reserved_mb, peek_mb = get_avg_time_and_memory(n, img_path)
            
            # 填充结果（保留4位小数，便于精准对比）
            df.loc[img_name, '平均耗时(ms)'] = round(avg_time * 1000, 4)
            df.loc[img_name, '平均已分配内存(MB)'] = round(avg_allocated_mb, 4)
            df.loc[img_name, '平均预留内存(MB)'] = round(avg_reserved_mb, 4)
            df.loc[img_name, 'CUDA峰值内存(含所有分配)(MB)'] = round(peek_mb, 4)
            
            print(f"图片 {img_name} 处理完成！")
        except Exception as e:
            print(f"处理图片 {img_name} 失败：{str(e)}")
        print("="*60)
    
    # 保存结果到CSV（便于后续分析）
    csv_filename = "gpu1_result_with_full_peek_memory.csv"
    df.to_csv(csv_filename, encoding='utf-8-sig')
    print(f"测试完成，结果已保存到 {csv_filename}")
    print("\n测试结果预览：")
    print("-"*60)
    print(df)
