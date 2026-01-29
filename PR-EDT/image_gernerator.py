import os
import numpy as np
from PIL import Image

def generate_binary_images(seed=42):

    os.makedirs("image", exist_ok=True)
    
    sizes = [1024, 2048, 4096, 8192]
    total_files = len(sizes) * 99

    rng = np.random.default_rng(seed)
    
    print(f"开始生成图片，共 {total_files} 张...")
    
    for size in sizes:
        print(f"\n正在生成 {size}x{size} 的图片...")
        
        for percent in range(1, 100):

            random_vals = rng.integers(0, 100, size=(size, size), dtype=np.uint8)
            binary_mask = (random_vals < percent).astype(np.uint8)
            
            img_array = (1 - binary_mask) * 255

            img = Image.fromarray(img_array, mode='L')
            filename = f"image/black{percent}_{size}x{size}.png"
            img.save(filename)
            
            if percent % 20 == 0:
                print(f"  已完成 {percent}% 的图片")
                
        print(f"  ✓ {size}x{size} 完成 (99张)")
    
    print(f"\n全部完成！图片保存在 ./image/ 文件夹中")

if __name__ == "__main__":
    generate_binary_images()
