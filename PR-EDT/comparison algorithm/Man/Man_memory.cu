#include <iostream>
#include <math.h>
#include <thrust/extrema.h>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime_api.h>

using namespace std;
using namespace cv;

// 函数前置声明：解决"identifier undefined"错误
void print_cuda_memory_usage(const std::string& step_info);
void print_cuda_peak_memory_usage();

// 辅助函数：查询当前CUDA设备内存使用情况（已用/剩余/总内存）
void print_cuda_memory_usage(const std::string& step_info) {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "Error querying CUDA memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    size_t used_mem = total_mem - free_mem;
    std::cout << "[" << step_info << "] "
              << "Total Memory: " << (total_mem / 1024 / 1024) << " MB, "
              << "Used Memory: " << (used_mem / 1024 / 1024) << " MB, "
              << "Free Memory: " << (free_mem / 1024 / 1024) << " MB" << std::endl;
}

// 辅助函数：记录CUDA峰值内存使用（兼容所有CUDA版本）
void print_cuda_peak_memory_usage() {
    size_t peak_mem_free, peak_mem_total;
    cudaError_t err = cudaMemGetInfo(&peak_mem_free, &peak_mem_total);
    if (err != cudaSuccess) {
        std::cerr << "Error querying CUDA peak memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    size_t peak_mem_used = peak_mem_total - peak_mem_free;
    std::cout << "[Peak Memory (Compatible Version)] " << (peak_mem_used / 1024 / 1024) << " MB" << std::endl;
}

// 数据打印
__global__ void cuda_test(unsigned char *g_arr, int cols, int rows) {
	int inx = threadIdx.x + blockDim.x * blockIdx.x;
	printf(" %d,%d  ", inx,g_arr[inx]);
}

typedef struct{
	int x;
	int y;
} GPoint;

#pragma region step01

// 图像打印
__global__ void img_log(unsigned char *nums, int cols, int rows){
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++)
			printf("%d ",nums[i * cols + j] == 0);
		printf("\n");
	}
}

// EDM算法步骤一
// 并行计算每层元素与其垂直向上方向最近黑点的距离 
__global__ void cuda_step1(int *nums,unsigned char *g_arr, int cols, int rows,int * pfixsum){

	//行
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int block = blockDim.x * gridDim.x;
	for(;i < cols;i += block){
		// 初始化每层元素距离顶部的距离
		for(int j = 0;j < rows;j++){
			nums[j * cols + i] = 9999999;
		}
		// 计算每层元素与其垂直向上方向最近黑点的距离 
		if(g_arr[i] == 0)
				nums[i] = 0;
		for(int j = 1;j < rows;j++){
			if(g_arr[j * cols + i] == 0)
				nums[j * cols + i] = 0;
			else 
				nums[j * cols + i] =  nums[(j - 1) * cols + i] + 1;
		}
		for(int j = rows - 2;j >= 0;j--)
			nums[j * cols + i] = min(nums[j * cols + i],nums[(j + 1) * cols + i] + 1);	
		pfixsum[i] = nums[i] < 9999999;
	}
	
}

// 前缀和计算步骤一
// 每个线程计算局部数据的前缀和，并将每组局部数据的前缀和结果存入全局内存g_temp中
__global__ void scan1(int * g_num, int * g_temp,int size){

    __shared__ int temp[512];

    int tx = threadIdx.x;

    int bx = blockIdx.x;

    int inx = bx * blockDim.x + tx;

    if(inx >= size)

        return;
	// 局部前缀和计算
    temp[tx] = g_num[inx];

    __syncthreads();

    for(int i = 1;i <= tx; i <<= 1){

        temp[tx] = g_num[inx] + g_num[inx - i];

        __syncthreads();

        g_num[inx] = temp[tx];

        __syncthreads();

    }

	//保存局部前缀和结果至全局内存g_temp中
    if(tx == 511){

        g_temp[bx] = g_num[inx];

    }

}

//前缀和计算步骤二
//计算全局内存g_tmp的前缀和
__global__ void scan2(int * g_temp,int size) {

    __shared__ int temp[512];

    int tx = threadIdx.x;

    int bx = blockIdx.x;

    int inx = bx * blockDim.x + tx;

    if(inx >= size)

        return;
	// 全局内存g_temp的前缀和计算
    temp[tx] = g_temp[inx];

    __syncthreads();

    for(int i = 1;i <= tx; i <<= 1){

        temp[tx] = g_temp[inx] + g_temp[inx - i];

        __syncthreads();

        g_temp[inx] = temp[tx];

        __syncthreads();

    }

}

//前缀和计算步骤三
//将g_tmp的结果累加到g_num中，获得全局的前缀和
__global__ void scan3(int * g_num, int * g_temp,int size){

    int inx = blockIdx.x * blockDim.x + threadIdx.x + 512;

    int threadSize = blockDim.x * gridDim.x;
	
    for(;inx < size;inx += threadSize){
            g_num[inx] += g_temp[inx / 512 - 1];

    }

}

__global__ void cuda_init(GPoint * g_point,int * step1_arr,int * pfixsum,int rows,int cols){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadSize = blockDim.x * gridDim.x;
	for(;inx / cols < rows;inx += threadSize){
        if(inx % cols == 0){
			if(pfixsum[0] == 1){
				g_point[inx].x = 0;
				g_point[inx].y = step1_arr[inx];
			}
		}
		else if(pfixsum[inx % cols - 1] != pfixsum[inx % cols]){
			g_point[inx / cols * cols + pfixsum[inx % cols] - 1].x = inx % cols;
			g_point[inx / cols * cols + pfixsum[inx % cols] - 1].y = step1_arr[inx];

		}
    }
}

#pragma endregion

#pragma region step02

__device__ int BinSearchLeftPoint(GPoint *g_point,int low,int mid,int high,int *g_arr_ignore,int xuhao,int dis);
__device__ int BinSearchRightPoint(GPoint *g_point,int low,int mid,int high,int *g_arr_ignore,int xuhao,int dis);
__device__ int BinSearchRightChuPoint(GPoint *g_point, int low, int high,int *g_arr_ignore, int xuhao,int leftMid);
__device__ int BinSearcLeftChuPoint(GPoint *g_point, int low, int high,int *g_arr_ignore, int xuhao,int rightMid);

//a,b两点的垂直平分线与x轴的交点
__device__ double calculateVerticalBisectorX(GPoint a,GPoint b){
	return (double)(a.y + b.y) / 2 * (double)(a.y - b.y) / (double)(a.x - b.x) + (double)(a.x + b.x) / 2;
}

// 基于二分搜索的Voronoi Diagram算法
// 通过左右两端的二分查找寻找左右分区的非屏蔽点的临界位置，得到本行有效的非屏蔽点数组
__device__ void binsearchVoronoiDiagram(GPoint * a,int a_l,int a_size,int b_l,int b_size,int * returnL,int * returnSize){
	int left = a_l + 1;
	int right = a_size - 1;
	int mid = a_l;
	bool flag = false;
	double x;
	int left1,right1,mid1;
	bool flag1;
	// 通过二分查找寻找左分区的非屏蔽点的临界位置 
	while(left <= right){
		flag = false;
		mid = (left + right) >> 1;
		x = calculateVerticalBisectorX(a[mid],a[mid - 1]); 
		left1 = b_l,right1 = b_size - 2;
		mid1 = b_l;
		flag1 = false;
		while(left1 <= right1){
			flag1 = false;
			mid1 = (left1 + right1) >> 1;

			(flag1 = (calculateVerticalBisectorX(a[mid],a[mid1]) >= calculateVerticalBisectorX(a[mid1],a[mid1 + 1]))) ? left1 = mid1 + 1 : right1 = mid1 - 1;
		}
		if(flag1)
			mid1++;
		(flag = (x >= calculateVerticalBisectorX(a[mid],a[mid1]))) ? right = mid - 1 : left = mid + 1;
		
	}
	int p = flag ? mid - 1 : mid;
		
	left = b_l;
	right = b_size - 2;
	flag = false;
	mid = b_l;
	// 通过二分查找寻找右分区非屏蔽点的临界位置 
	while(left <= right){
		flag = false;
		mid = (left + right) >> 1;
		x = calculateVerticalBisectorX(a[mid],a[mid + 1]);
		left1 = a_l + 1,right1 = a_size - 1;
		mid1 = a_l;
		flag1 = false;
		while (left1 <= right1){
			flag1 = false;
			mid1 = (left1 + right1) >> 1;
			(flag1 = (calculateVerticalBisectorX(a[mid1],a[mid]) <= calculateVerticalBisectorX(a[mid1 - 1],a[mid1]))) ? right1 = mid1 - 1 : left1 = mid1 + 1;
		}
		if(flag1)
			mid1--;
		(flag = (x <= calculateVerticalBisectorX(a[mid1],a[mid]))) ? left = mid + 1 : right = mid - 1;
		
	}
	int q = flag ? mid + 1 : mid;
	*returnL = a_l;
	*returnSize = p + 1 + b_size - q;
	for(int i = q;i < b_size;i++)
		a[++p]= a[i]; 
}

// 归并算法
__device__ void MergePass(GPoint *g_point,int low,int high,int * l,int *length){

	// 原数组可二分情况 
	if(low < high){
		// 获取左分区 
		int a_l = 0;
		int a_size = 0;
		MergePass(g_point,low,(low + high) >> 1,&a_l,&a_size);
		// 获得右分区
		int b_l = 0;
		int b_size = 0; 
		MergePass(g_point,((low + high) >> 1) + 1,high,&b_l,&b_size);
		// 返回合并左右分区后的结果 
		binsearchVoronoiDiagram(g_point,a_l,a_size,b_l,b_size,l,length);
	}
	else{ // 原数组不可继续二分 直接返回 
		*length = low + 1;
		*l = low;
	}
}

// EDM算法步骤二
// 压缩每行的黑点，并通过基于归并排序的Voronoi Diagram算法计算出每行有效的非屏蔽点数组
 __global__ void cuda_step2(GPoint *g_point,int *g_arr_step1, int cols,int rows, int *length,int n){
 	int inx = threadIdx.x + blockDim.x * blockIdx.x;
 	int block = blockDim.x * gridDim.x;
	//压缩黑点
 	for(inx *= n;inx < rows; inx+=block*n ){
 		int size = 0;
 		for(int i = 0; i < cols; i++){
 			if(g_arr_step1[inx * cols + i] < 9999999){
 				g_point[inx *cols + size].x = i;
 				g_point[inx *cols + size++].y = g_arr_step1[inx*cols+i];
 			}
 		}
 		int l;
		//调用基于归并排序的Voronoi Diagram算法计算出每行有效的非屏蔽点数组
 		MergePass(g_point,inx*cols,inx*cols+size-1,&l,&length[inx]);
 		length[inx] -= inx * cols;
 	}
 }

// 二分搜索寻找左右分区的非屏蔽点的临界位置
__device__ int binsearch(GPoint * g_point,int size,GPoint target){
	int left = 1,right = size,mid = size;
	while(left < right){
		mid = (left + right) >> 1;
		if(calculateVerticalBisectorX(g_point[mid - 1],g_point[mid]) >= calculateVerticalBisectorX(g_point[mid],target))
			right = mid;
		else
			left = mid + 1;
	}
	return right;
} 

__global__ void cuda_step4(GPoint *g_point,GPoint * temp_point,int cols,int rows, int *length,int *pfix,int *temp,bool *flag){
	int inx = threadIdx.x;
	int blockSize = gridDim.x;
	int bix = blockIdx.x;
	int threadSize = blockDim.x;
	int size;
	for(;bix < rows; bix+=blockSize ){
		if(inx == 0){
			pfix[bix * cols] = 1;
			flag[bix] = true;
		}
		__syncthreads();
		size = length[bix] - 1;
		while(flag[bix]){
			if(inx == 0){
				pfix[bix * cols + size] = 1;
				flag[bix] = false;
			}
			__syncthreads();
			for(int i = inx + 1;i < size;i += threadSize){
				if(calculateVerticalBisectorX(g_point[bix * cols + i - 1],g_point[bix * cols + i]) >= calculateVerticalBisectorX(g_point[bix * cols + i],g_point[bix * cols + i + 1])){
					flag[bix] = true;
					pfix[bix * cols + i] = 0;
				}
				else
					pfix[bix * cols + i] = 1;			
			}
			__syncthreads();
			for(int j = 1;j <= size;j <<= 1){
				for(int k = size - inx;k >= j;k -= threadSize){
					temp[bix * cols + k] = pfix[bix * cols + k] + pfix[bix * cols + k - j];
				}
				__syncthreads();
				for(int k = size - inx;k >= j;k -= threadSize){
					pfix[bix * cols + k] = temp[bix * cols + k];
				}
			}
			__syncthreads();	
			for(int i = inx + 1;i <= size;i += threadSize){
				if(pfix[bix * cols + i] != pfix[bix * cols + i - 1]){
					temp_point[bix * cols + pfix[bix * cols + i - 1]] = g_point[bix * cols + i]; 
				}
			}
			__syncthreads();
			size = pfix[bix * cols + size] - 1;
			for(int i = inx + 1;i <= size;i += threadSize){
				g_point[bix * cols + i] = temp_point[bix * cols + i];
			}
			__syncthreads();
		}
		if(inx == 0)
			length[bix] = size + 1;
		
	}
}

__global__ void step2_init(GPoint *g_point,int *g_arr_step1, int cols,int rows, int *length){
	int inx = threadIdx.x + blockDim.x * blockIdx.x;
	int block = blockDim.x * gridDim.x;
	int size;
	for(;inx < rows;inx += block){
		size = 0;
		for(int j = 0;j < cols;j++){
			if(g_arr_step1[inx * cols + j] < 9999999){
				g_point[inx * cols + size].x = j;
				g_point[inx * cols + size].y = g_arr_step1[inx * cols + j];
				size++;
			}
		}
		length[inx] = size;
	}
}

__global__ void cuda_step21(int cols,int rows, int length,int *size){
	int inx = threadIdx.x;
	int bix = blockIdx.x;
	int threadSize = blockDim.x;
	int blockSize = gridDim.x;
	for(int i = bix;i < rows; i += blockSize){
		for(int j = inx;j < length;j += threadSize){
			size[i * cols + j] = min(2,length - j);
		}
	}
}

__global__ void cuda_step22(int cols,int rows, int *length,int *size){
	int inx = threadIdx.x + blockDim.x * blockIdx.x;
	int block = blockDim.x * gridDim.x;
	for(;inx < rows;inx += block){
		length[inx] = size[inx * cols];
	}
}

__global__ void cuda_step23(GPoint *g_point,int cols,int rows, int *length,int *size){
	int inx = threadIdx.x;
	int bix = blockIdx.x;
	int threadSize = blockDim.x;
	int blockSize = gridDim.x;
	int index1,index2;
	int t;
	for(int i = bix;i < rows; i += blockSize){
		for(int j = 4;j / 2 < length[i]; j *= 2){
			__syncthreads();	
			for(int k = inx;k * j + j / 2 < length[i]; k += threadSize){
				index1 = k * j;
				binsearchVoronoiDiagram(&g_point[i * cols + index1],0,size[i * cols + index1],j / 2,j / 2 + size[i * cols + index2],&t,&size[i * cols + index1]);

			}
									
		}
	}
}

__global__ void cuda_step231(GPoint *g_point,int cols,int rows, int length,int *size,int j,int n){
	int inx = threadIdx.x + blockDim.x * blockIdx.x;
	int threadSize = blockDim.x * gridDim.x;
	for(int i = inx;i / length < rows;i += threadSize){
		binsearchVoronoiDiagram(&g_point[i / length * cols * n + i % length * j],0,size[i / length * cols * n + i % length * j],j / 2,j / 2 + size[i / length * cols * n + i % length * j + j / 2],&size[i / length * cols * n + i % length * j],&size[i / length * cols * n + i % length * j]);
	}
}

__global__ void fun1111(GPoint* nums,int cols,int rows,int length,int* lpoint,int* rpoint,int* temp,double* l,double* r){
	int bix = blockIdx.x;
	int inx = threadIdx.x;
	int blockSize = gridDim.x;
	int threadSize = blockDim.x;
	for(int i = bix;i < rows;i += blockSize){
		if(inx == 0){
			lpoint[bix * length] = -1;
			l[i * length] = 0;
		}
		for(int j = inx + 1;j < length;j += threadSize){
			lpoint[bix * length + j] = j - 1;
			l[i * length + j] = calculateVerticalBisectorX(nums[i * cols + j - 1],nums[i * cols + j]);
		}
		__syncthreads();
		for(int k = 2; k < length; k <<= 1){
			for(int j = length - inx - 1;j >= k; j -= threadSize){
				temp[bix * length + j] = lpoint[bix * length + j];
				if(calculateVerticalBisectorX(nums[i * cols + j - k],nums[i * cols + j]) >= l[i * length + j]){
					l[i * length + j] = calculateVerticalBisectorX(nums[i * cols + j - k],nums[i * cols + j]);
					temp[bix * length + j] = j - k;
				}
				if(lpoint[bix * length + j - k] != -1 && calculateVerticalBisectorX(nums[i * cols + lpoint[bix * length + j - k]],nums[i * cols + j]) >= l[i * length + j]){
					l[i * length + j] = calculateVerticalBisectorX(nums[i * cols + lpoint[bix * length + j - k]],nums[i * cols + j]);
					temp[bix * length + j] = lpoint[bix * length + j - k];
				}
			}
			__syncthreads();

			for(int j = length - inx - 1;j >= k; j -= threadSize){
				lpoint[bix * length + j] = temp[bix * length + j];
			}
			__syncthreads();

		}

		if(inx == 0){
			rpoint[bix * length + length - 1] = -1;
			r[i * length + length - 1] = cols - 1;
		}
		for(int j = inx;j < length - 1;j += threadSize){
			rpoint[bix * length + j] = j + 1;
			r[i * length + j] = calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + j + 1]);
		}
		__syncthreads();
		for(int k = 2; k < length; k <<= 1){
			for(int j = inx;j + k < length; j += threadSize){
				temp[bix * length + j] = rpoint[bix * length + j];
				if(calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + j + k]) <= r[i * length + j]){
					r[i * length + j] = calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + j + k]);
					temp[bix * length + j] = j + k;
				}
				if(rpoint[bix * length + j + k] != -1 && calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + rpoint[bix * length + j + k]]) <= r[i * length + j]){
					r[i * length + j] = calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + rpoint[bix * length + j + k]]);
					temp[bix * length + j] = rpoint[bix * length + j + k];
				}
			}
			__syncthreads();

			for(int j = inx;j + k < length; j += threadSize){
				rpoint[bix * length + j] = temp[bix * length + j];
			}
			__syncthreads();

		}
		__syncthreads();
	}
}

#pragma endregion

__device__ double disancePoint(GPoint a,GPoint b){
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

#pragma region step03
//第三步
__global__ void cuda_step3(GPoint *g_arr_step3_point,GPoint *g_arr_step2,int *length,int cols,int rows,int *res){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int bolck = blockDim.x * gridDim.x;
	
	for(;i < rows; i+= bolck){
		int left = 0;
		int right;
		for(int j = 0;j < length[i] - 1;j++){
			right = min(cols - 1,(int)floor(calculateVerticalBisectorX(g_arr_step2[i * cols + j],g_arr_step2[i * cols + j + 1])));
			while(left <= right){
					res[i * cols + left] = (g_arr_step2[i * cols + j].y) * (g_arr_step2[i * cols + j].y) + (left - g_arr_step2[i * cols + j].x) * (left - g_arr_step2[i * cols + j].x);
				left++;
				
			}
		}
		right = cols - 1;
		while(left <= right){
					res[i * cols + left] = (g_arr_step2[i * cols + length[i] - 1].y) * (g_arr_step2[i * cols + length[i] - 1].y) + (left - g_arr_step2[i * cols + length[i] - 1].x) * (left - g_arr_step2[i * cols + length[i] - 1].x);
			left++;
		}	
	}
	
}

__global__ void cuda_step31(GPoint *g_arr_step2,int * len,int rows,int cols,int *res){
	int inx = threadIdx.x;
	int bolckSize = gridDim.x;
	int bix = blockIdx.x;
	int threadSize = blockDim.x;
	int x1,y1;
	int left,right;
	for(;bix < rows; bix += bolckSize){
		for(int i = inx;i < len[bix];i += threadSize){
			x1 = g_arr_step2[bix * cols + i].x;
			y1 = g_arr_step2[bix * cols + i].y;
			left = i == 0 ? 0 : max(0,(int)ceil(calculateVerticalBisectorX(g_arr_step2[bix * cols + i - 1],g_arr_step2[bix * cols + i])));
			right = i == len[bix] - 1 ? cols - 1 : min(cols - 1,(int)floor(calculateVerticalBisectorX(g_arr_step2[bix * cols + i],g_arr_step2[bix * cols + i + 1])));
			while(left <= right){
				res[bix * cols + left] = y1 * y1 + (x1 - left) * (x1 - left);
				left++;
			}
		}

	}
}

__global__ void cuda_step32(GPoint *g_arr_step2,int len,double* l,double* r,int rows,int cols,int *res){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int bolck = blockDim.x * gridDim.x;
	int left,right;
	int x1,y1;
	int y,x;
	for(;i < rows * len; i += bolck){
		y = i / len;
		x = i % len;
		x1 = g_arr_step2[y * cols + x].x;
		y1 = g_arr_step2[y * cols + x].y;
		left = max(0,(int)ceil(l[i]));
		right = min(cols - 1,(int)floor(r[i]));
		while(left <= right){
			res[y * cols + left] = y1 * y1 + (x1 - left) * (x1 - left);
			left++;
		}
	}
	
}

__global__ void cuda_temp(int *g_arr_step4,int size){
	int inx = threadIdx.x + blockDim.x * blockIdx.x;
	int block = blockDim.x * gridDim.x;

	for(int i = inx; i < size ; i+=block ){
		g_arr_step4[i] = 8192*8192*4;
	}

}
#pragma endregion

__global__ void BF2(int * up,int * down,int * left,int * right,double * res,int size,int cols,int rows){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int Size = gridDim.x * blockDim.x;
	double distance;
	int y,x,index,y1,x1;
	int start,end,r;
	for(;i < size;i += Size){
		y = i / cols;
		x = i % cols;
		index = y * cols;
		distance = 8192 * 8192 + 100;
		start = 0;
		end = cols;
		if(up[i] != -1){
			r = min(abs(up[i] - y),abs(down[i] - y));
			start = max(0,x - r);
			end = min(cols,x + r + 1);
		}
		if(left[i] != -1){
			r = min(abs(left[i] - x),abs(right[i] - x));
			start = max(0,x - r);
			end = min(cols,x + r + 1);
		}
		// up
		for(int j = start;j < end;j++){
			y1 = up[index + j];
			if(y1 != -1)
				distance = min(distance,sqrt(double((y - y1) * (y - y1) + (x - j) * (x - j))));
		}

		// down
		for(int j = start;j < end;j++){
			y1 = down[index + j];
			if(y1 != -1)
				distance = min(distance,sqrt(double((y - y1) * (y - y1) + (x - j) * (x - j))));
			
		}
		res[i] = distance;
	}
}

double* fun4(vector<vector<bool>>& nums){
	int size = nums.size() * nums[0].size();
	int * up = (int*) malloc(sizeof(int) * size);
	int * down = (int*) malloc(sizeof(int) * size);
	int * left = (int*) malloc(sizeof(int) * size);
	int * right = (int*) malloc(sizeof(int) * size);
	
	up[0] = nums[0][0] ? 0 : -1;
	left[0] = nums[0][0] ? 0 : -1;
	for(int j = 1;j < nums[0].size();j++) {
		if(nums[0][j]){
			up[j] = 0;
			left[j] = j;
		}else{
			up[j] = -1;
			left[j] = left[j - 1];
		}
	}
	for(int i = 1;i < nums.size();i++){
		left[i * nums[i].size()] = nums[i][0] ? 0 : -1;
		up[i * nums[i].size()] = nums[i][0] ? i : up[(i - 1) * nums[i].size()];
		for(int j = 1;j < nums[i].size();j++){
			if(nums[i][j]){
				up[i * nums[i].size() + j] = i;
				left[i * nums[i].size() + j] = j;
			}else{
				up[i * nums[i].size() + j] = up[(i - 1) * nums[i].size() + j];
				left[i * nums[i].size() + j] = left[i * nums[i].size() + j - 1];
			}
		}
	}

	down[size - 1] = nums.back().back() ? nums.size() - 1 : -1;
	right[size - 1] = nums.back().back() ? nums[0].size() - 1 : -1;
	for(int j = nums[0].size() - 2;j >= 0;j--){
		if(nums.back()[j]){
			down[(nums.size() - 1) * nums[0].size() + j] = nums.size() - 1;
			right[(nums.size() - 1) * nums[0].size() + j] = j;
		}else{
			down[(nums.size() - 1) * nums[0].size() + j] = -1;
			right[(nums.size() - 1) * nums[0].size() + j] = right[(nums.size() - 1) * nums[0].size() + j + 1];
		}
		
	}
	
	for(int i = nums.size() - 2;i >= 0;i--){
		right[(i + 1) * nums[i].size() - 1] = nums[i].back() ? nums[i].size() - 1 : -1;
		down[(i + 1) * nums[i].size() - 1] = nums[i].back() ? i : down[(i + 2) * nums[i].size() - 1];
		for(int j = nums[i].size() - 2;j >= 0;j--){
			if(nums[i][j]){
				down[i * nums[i].size() + j] = i;
				right[i * nums[i].size() + j] = j;
			}else{
				down[i * nums[i].size() + j] = down[(i + 1) * nums[i].size() + j];
				right[i * nums[i].size() + j] = right[i * nums[i].size() + j + 1];
			}
		}
	}
	int * UP = NULL;
	int * DOWN = NULL;
	int * LEFT = NULL;
	int * RIGHT = NULL;
	double * res = NULL;
	cudaMalloc((void**)& UP,sizeof(int) * size);
	cudaMalloc((void**)& DOWN,sizeof(int) * size);
	cudaMalloc((void**)& LEFT,sizeof(int) * size);
	cudaMalloc((void**)& RIGHT,sizeof(int) * size);
	cudaMalloc((void**)& res,sizeof(double) * size);
	cudaMemcpy(UP,up,sizeof(int) * size,cudaMemcpyHostToDevice);
	cudaMemcpy(DOWN,down,sizeof(int) * size,cudaMemcpyHostToDevice);
	cudaMemcpy(LEFT,left,sizeof(int) * size,cudaMemcpyHostToDevice);
	cudaMemcpy(RIGHT,right,sizeof(int) * size,cudaMemcpyHostToDevice);

	cudaEvent_t start_event, stop_event;
	float time_kernel; 
	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	BF2<<<32,512>>>(UP,DOWN,LEFT,RIGHT,res,size,int(nums[0].size()),int(nums.size()));
	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
	printf("BF2 time %fus\n", time_kernel * 1000);
	double * res1 = (double*) malloc(sizeof(double) * size);
	cudaMemcpy(res1,res,sizeof(double) * size,cudaMemcpyDeviceToHost);
	return res1;
}

// 暴力算法 
double* fun1(bool* flag,int cols,int rows){
	GPoint* nums1 = (GPoint*)malloc(sizeof(GPoint) * cols * rows); // 黑点集合 
	GPoint* nums2 = (GPoint*)malloc(sizeof(GPoint) * cols * rows); // 白点集合
	int nums1_size = 0;
	int nums2_size = 0; 
	double* res = (double*)malloc(sizeof(double) * cols * rows);
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++)
			if(flag[i * cols + j]){
				nums1[nums1_size].x = j;
				nums1[nums1_size++].y = i;
				res[i * cols + j] = 0;
			}
			else{
				nums2[nums2_size].x = j;
				nums2[nums2_size++].y = i;
				res[i * cols + j] = 99999999;
			}
	}
	int y,x;
	for(int i = 0;i < nums2_size;i++){
		y = nums2[i].y;
		x = nums2[i].x;
		for(int j = 0;j < nums1_size;j++){
			if(res[y * cols + x] > sqrt((y - nums1[j].y) * (y - nums1[j].y) + (x - nums1[j].x) * (x - nums1[j].x))){
				res[y * cols + x] = sqrt((y - nums1[j].y) * (y - nums1[j].y) + (x - nums1[j].x) * (x - nums1[j].x));
			}
		}
			
	}
	free(nums1);
	free(nums2);
	return res;
}

#pragma endregion

void tranformation(vector<vector<bool>>& flag) {
    int rows = flag.size();
    int cols = flag[0].size();
    uint64_t t1, t2;
    cudaEvent_t start_event, stop_event;
    float time_kernel; 
    float time_kernel_total = 0; 
    int block = 32;
    int index = 512;

    // 初始内存状态
    print_cuda_memory_usage("Init (Before Malloc)");

    // 第一步：内存分配
    unsigned char* output_data = (unsigned char*)malloc(sizeof(unsigned char) * rows * cols);
    for(int i = 0;i < rows;i++){
        for(int j = 0;j < cols;j++){
            output_data[i * cols + j] = flag[i][j] ? 0 : 255;
        }
    }
    unsigned char* g_output_data;
    cudaMalloc((void**)&g_output_data,sizeof(unsigned char) * rows * cols);
    cudaMemcpy(g_output_data,output_data,sizeof(unsigned char) * rows * cols,cudaMemcpyHostToDevice);

    int *arr_step1 = (int *)malloc(rows*cols*sizeof(int));
    int *g_arr_step1 = NULL;
    cudaMalloc((void **)&g_arr_step1,rows*cols*sizeof(int));
    int * pfixsum;
    int * pfixsum_temp;
    cudaMalloc((void **)&pfixsum,cols*sizeof(int));
    cudaMalloc((void **)&pfixsum_temp,16*sizeof(int));

    // 第一步内存分配完成后
    print_cuda_memory_usage("Step 1 (After Malloc)");

    // 第一步执行
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);

    cuda_step1<<<block , index>>>(g_arr_step1,g_output_data, cols, rows,pfixsum);	
    scan1<<<block,index>>>(pfixsum,pfixsum_temp,cols);
    scan2<<<1,index>>>(pfixsum_temp,16);
    scan3<<<block,index>>>(pfixsum,pfixsum_temp,cols);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    time_kernel_total += time_kernel;
    cudaMemcpy(arr_step1,g_arr_step1,rows * cols * sizeof(int),cudaMemcpyDeviceToHost);

    // 第二步：内存分配
    GPoint *g_arr_step2 = NULL;
    int *length = NULL;
    cudaMalloc((void **)&length,rows*sizeof(int));
    cudaMalloc((void **)&g_arr_step2,rows*cols*sizeof(GPoint));

    int* lpoint;
    cudaMalloc((void **)&lpoint,block * cols * sizeof(int));
    int* rpoint;
    cudaMalloc((void **)&rpoint,block * cols * sizeof(int));
    double* l;
    cudaMalloc((void **)&l,rows * cols * sizeof(double));
    double* r;
    cudaMalloc((void **)&r,rows * cols * sizeof(double));
    int* temp;
    cudaMalloc((void **)&temp,block * cols * sizeof(int));

    // 第二步内存分配完成后
    print_cuda_memory_usage("Step 2 (After Malloc)");

    // 第二步执行
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);

    int len;
    cudaMemcpy(&len,pfixsum + cols - 1,sizeof(int),cudaMemcpyDeviceToHost);
    cuda_step2<<<64,512>>>(g_arr_step2,g_arr_step1,cols,rows,length,1);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    time_kernel_total += time_kernel;

    // 第三步：内存分配
    int *res = NULL;
    cudaMalloc((void **)&res,rows*cols*sizeof(int));
    cuda_temp<<<8,512>>>(res,rows*cols);
    int *arr_step4 = (int *)malloc(rows*cols*sizeof(int));
    GPoint *arr_step3 = (GPoint *)malloc(rows*cols*sizeof(GPoint));
    GPoint *g_arr_step3_point = NULL;
    cudaMalloc((void **)&g_arr_step3_point,rows*cols*sizeof(GPoint));

    // 第三步内存分配完成后
    print_cuda_memory_usage("Step 3 (After Malloc)");

    // 第三步执行
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);

    cuda_step31<<<block,index>>>(g_arr_step2,length,rows,cols,res);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    time_kernel_total += time_kernel;

    cudaMemcpy(arr_step3,g_arr_step3_point,rows * cols * sizeof(GPoint),cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_step4,res,rows * cols * sizeof(int),cudaMemcpyDeviceToHost);

    // 释放前内存状态
    print_cuda_memory_usage("Before Free (After Step 3)");

    // 内存释放
    cudaFree(g_arr_step1);
    cudaFree(length);
    cudaFree(g_arr_step2);
    cudaFree(g_arr_step3_point);
    cudaFree(res);
    cudaFree(g_output_data);
    cudaFree(lpoint);
    cudaFree(rpoint);
    cudaFree(l);
    cudaFree(r);
    cudaFree(temp);
    cudaFree(pfixsum);
    cudaFree(pfixsum_temp);
    cudaDeviceSynchronize();

    // 释放后内存状态 & 峰值内存
    print_cuda_memory_usage("After Free");
    print_cuda_peak_memory_usage();

    free(arr_step3);
    free(arr_step4);
    free(output_data);
    free(arr_step1);
    printf("Total kernel time: %f us\n", time_kernel_total * 1000);
}

int main(int argc, char **argv) {
    size_t limit = 4096;
    cudaDeviceSetLimit(cudaLimitStackSize,limit);

    // 定义图片列表（同级image目录下）
    vector<string> image_files = {
        "image/L-8192.png",
    	"image/L-8192.png",
        "image/L-4096.png",
        "image/L-2048.png",
        "image/L-1024.png",
	"image/L-1024.png"
    };

    vector<vector<bool>> flag;
    for (const string& img_path : image_files) {
        // 1. 读取图片（灰度图模式）
        Mat img = imread(img_path, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Error: Cannot read image " << img_path << endl;
            continue;
        }
        cout << "=====================================" << endl;
        cout << "Processing image: " << img_path << endl;
        cout << "Image size: " << img.rows << " x " << img.cols << endl;
        cout << "=====================================" << endl;

        // 2. 转换为vector<vector<bool>> flag
        flag.clear();
        flag.resize(img.rows);
        for (int i = 0; i < img.rows; i++) {
            flag[i].resize(img.cols);
            for (int j = 0; j < img.cols; j++) {
                // 灰度图中，0为黑（对应flag=true），255为白（对应flag=false）
                flag[i][j] = (img.at<unsigned char>(i, j) == 0);
            }
        }

        // 3. 调用EDM算法，测量性能和内存
        tranformation(flag);
        cout << "-------------------------------------" << endl;
        cout << "Image " << img_path << " processed completed." << endl;
        cout << "-------------------------------------" << endl;
    }

    // 重置CUDA设备，释放剩余资源
    cudaDeviceReset();
    return 0;
}
