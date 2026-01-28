# 1. 配置CUDA安装路径（默认/usr/local/cuda，若你的路径不同请修改）
CUDA_PATH ?= /usr/local/cuda
CUDA_BIN = $(CUDA_PATH)/bin
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# 2. 配置编译器和链接器（使用nvcc作为主编译器，匹配给定命令）
NVCC = $(CUDA_BIN)/nvcc

# 3. 配置OpenCV路径（参考给定的nvcc命令）
OPENCV_INC = -I"/usr/local/include/opencv4"  # OpenCV头文件路径
OPENCV_LIB = -L"/usr/local/lib"             # OpenCV库文件路径

# 4. 编译选项（匹配给定的nvcc命令，拆分调试选项、核心选项）
# 调试选项：-G -g （启用CUDA调试模式，发布版本可注释这两个参数）
NVCC_DEBUG_FLAGS = -G -g
# 核心选项：
# -rdc=true：启用CUDA设备代码重定位（必须，参考给定命令）
# -std=c++11：C++11标准
# -arch=sm_75：显卡架构（根据你的显卡调整，同之前说明）
NVCC_CORE_FLAGS = -rdc=true -std=c++11 -arch=sm_75
# 整合所有编译选项（包含CUDA头文件、OpenCV头文件）
NVCC_FLAGS = $(NVCC_DEBUG_FLAGS) \
             $(NVCC_CORE_FLAGS) \
             -I$(CUDA_INC) \
             $(OPENCV_INC)

# 5. 链接选项（参考给定的nvcc命令，整合所有库路径和链接库）
# CUDA链接库：cudart（CUDA运行时库）、cuda（CUDA核心库）
# OpenCV链接库：参考给定命令，完整保留
LD_FLAGS = -L$(CUDA_LIB) \
           $(OPENCV_LIB) \
           -lcudart \
           -lcuda \
           -lopencv_core \
           -lopencv_imgcodecs \
           -lopencv_cudaimgproc \
           -lopencv_cudaarithm

# 6. 目标文件和可执行文件（匹配给定命令的输出文件名edm）
TARGET = Man
SRC = edm.cu
OBJ = $(SRC:.cu=.o)

# 7. 默认目标：编译可执行文件
all: $(TARGET)

# 8. 编译CUDA文件（.cu -> .o）：使用整合后的编译选项
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 9. 链接生成可执行文件：使用编译选项+链接选项，匹配给定命令
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(TARGET) $(LD_FLAGS)

# 10. 清理编译产物（可执行文件、目标文件、调试临时文件）
clean:
	rm -rf $(TARGET) $(OBJ) *.o core.* *.dbg

# 11. 伪目标声明（避免与同名文件冲突）
.PHONY: all clean
