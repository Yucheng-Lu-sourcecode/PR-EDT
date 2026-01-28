#include <iostream>
#include <stdio.h>
#include <math.h>
#include <thrust/extrema.h>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

#include <time.h>
#include <sys/time.h>

using namespace cv;
using namespace cv::cuda;
using namespace std;

struct GPoint{
	int x;
	int y;
};

//不用动，已经调用了 -- 计时函数设置，可以精确到微秒us
namespace cci {
	namespace common {
		class event {
		public:
			static inline long long timestampInUS() {
				struct timeval ts;
				gettimeofday(&ts, NULL);
				return (ts.tv_sec * 1000000LL + (ts.tv_usec));
			};
		};
	}
}


// 将图片倒置
/// @brief 
/// @param g_input 	像素数组
/// @param rows 	行数
/// @param cols 	列数
/// @return 
__global__ void tan1(uchar* g_input,int rows,int cols){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	int size = rows / 2 * cols;
	int x,y;
	uchar temp;
	for(;inx < size;inx += threadSize){
		y = inx / cols;
		x = inx % cols;
		temp = g_input[(rows - 1 - y) * cols + x];
		g_input[(rows - 1 - y) * cols + x] = g_input[inx];
		g_input[inx] = temp;
	}
}

// 将点数组倒置
/// @brief 
/// @param g_point 	点数组
/// @param rows 	行数
/// @param cols 	列数
/// @return 
__global__ void tan2(GPoint* g_point,int rows,int cols){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	int size = rows / 2 * cols;
	int x,y;
	GPoint temp;
	for(;inx < size;inx += threadSize){
		y = inx / cols;
		x = inx % cols;
		temp = g_point[(rows - 1 - y) * cols + x];
		g_point[(rows - 1 - y) * cols + x] = g_point[inx];
		g_point[inx] = temp;
	}
}

//将记录数组长度数组倒置
/// @brief 
/// @param len 	记录数组长度数组
/// @param rows 行数
/// @return 
__global__ void tan3(int* len,int rows){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	int size = rows >> 1;
	int temp;
	for(;inx < size;inx += threadSize){
		temp = len[rows - 1 - inx];
		len[rows - 1 - inx] = len[inx];
		len[inx] = temp;
	}
}
// 计算两点中垂线与x轴交点

/// @brief 
/// @param a 	点a
/// @param b 	点b
/// @return 	a与b中垂线与x轴交点
__device__ double calculateVerticalBisectorX(GPoint a,GPoint b){
	return (double)(a.y + b.y) / 2 * (double)(a.y - b.y) / (double)(a.x - b.x) + (double)(a.x + b.x) / 2;
}
// 计算每个像素点上方最近黑点的距离
/// @brief 
/// @param g_input 	像素数组
/// @param nums 	点数组
/// @param rows 	行数
/// @param cols 	列数
/// @return 
__global__ void step1(uchar* g_input,GPoint * nums,int rows,int cols){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	for(;inx < cols;inx += threadSize){
		nums[inx].y = g_input[inx] * 99999;
		for(int i = 1;i < rows;i++){
			nums[i * cols + inx].y = nums[(i - 1) * cols + inx].y * (g_input[i * cols + inx] == 255) + (g_input[i * cols + inx] == 255);
		}
	}
}
// 计算每个块的第一行的邻近点，将剩余行的黑点压缩至数组末尾(每行单线程执行)
/// @brief 
/// @param nums 	点数组
/// @param rows 	行数
/// @param cols 	列数
/// @param n 		分块大小
/// @param length 	点数组长度数组
/// @return 
__global__ void step2(GPoint * nums,int rows,int cols,int n,int * length){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	int size;
	GPoint num;
	for(;inx < rows; inx+= threadSize){
		if(inx % n){
			size = cols;
			for(int i = cols - 1;i >= 0;i--){
				if(nums[inx * cols + i].y == 0){
					size--;
					nums[inx * cols + size].y = 0;
					nums[inx * cols + size].x = i;
				}
			}
			
		}else{
			size = 0;
			for(int i = 0; i < cols; i++){
				if(nums[inx * cols + i].y  < 99999){
					num.x = i;
					num.y = nums[inx * cols + i].y;
					while(size >= 2 && calculateVerticalBisectorX(nums[inx * cols + size - 2],nums[inx * cols + size - 1]) >= calculateVerticalBisectorX(nums[inx * cols + size - 1],num)){
						size--;
					}
					nums[inx * cols + size] = num;
					size++;
				}
			}
		}
		length[inx] = size;
	}
}

//计算每个块的第一行的邻近点，将剩余行的黑点压缩至数组末尾(抽样优化，每行多线程并行执行)
/// @brief 
/// @param nums		点数组 
/// @param rows 	行数
/// @param cols 	列数
/// @param n 		分块大小
/// @param r 		抽样分区终点
/// @param l 		抽样分区起始点
/// @param pfix 	前缀和数组
/// @return
__global__ void step2_1(GPoint * nums,int rows,int cols,int n,int * r,int* l,int * pfix){
	int inx = threadIdx.x;
	int threadSize = blockDim.x;
	int bix = blockIdx.x;
	int blockSize = gridDim.x;
	int m;
	GPoint num;
	for(;bix < rows; bix += blockSize){
		if(bix % n){
			for(int i = inx;i < cols;i += threadSize){
				pfix[bix * cols + i] = nums[bix * cols + i].y == 0;
			}
			__syncthreads();
			for(int j = 1;j < cols;j <<= 1){
				for(int k = inx;k + j < cols;k += threadSize){
					nums[bix * cols + k].x = pfix[bix * cols + k] + pfix[bix * cols + k + j]; 
				}
				__syncthreads();
				for(int k = inx;k + j < cols;k += threadSize){
					pfix[bix * cols + k] = nums[bix * cols + k].x;
				}
				__syncthreads();
			}
			for(int i = inx;i < cols - 1;i += threadSize){
				if(pfix[bix * cols + i] != pfix[bix * cols + i + 1]){
					nums[bix * cols + cols - pfix[bix * cols + i]].x = i;
          nums[bix * cols + cols - pfix[bix * cols + i]].y = 0;
				}
			}

			if(inx == 0){
				if(pfix[bix * cols + cols - 1])
					nums[bix * cols + cols - 1].x = cols - 1;
				r[bix] = cols - pfix[bix * cols];
			}

		}
		else{
			if(inx == 0){
				l[bix] = 0;
				r[bix] = -1;
			}
			for(int i = 0; i < cols; i++){
			__syncthreads();
				if(nums[bix * cols + i].y  < 99999){
					num.x = i;
					num.y = nums[bix * cols + i].y;
					if(r[bix] < 1){
						__syncthreads();
						if(inx == 0){
							r[bix]++;
							nums[bix * cols + r[bix]] = num;
						} 
					}
					else if(calculateVerticalBisectorX(nums[bix * cols],nums[bix * cols + 1]) >= calculateVerticalBisectorX(nums[bix * cols + 1],num)){
						__syncthreads();
						if(inx == 0){
							r[bix] = 1;
							nums[bix * cols + 1] = num;
						}
					}else if(calculateVerticalBisectorX(nums[bix * cols + r[bix] - 1],nums[bix * cols + r[bix]]) < calculateVerticalBisectorX(nums[bix * cols +  r[bix]],num)){
						__syncthreads();
						if(inx == 0){
							r[bix]++;
							nums[bix * cols + r[bix]] = num;
						}
					}else{
			
						if(inx == 0){
							l[bix] = 0;
						}
						__syncthreads();
						
						while(r[bix] - l[bix] > 2){
			
							m = (int)ceil((double)(r[bix] - l[bix]) / threadSize);
							pfix[bix * cols + min(inx,cols - 1)] = 1;
							if(inx == 0){
								pfix[bix * cols + min(cols - 1,threadSize)] = 1;
							}
				
							__syncthreads();
							if(l[bix] + inx * m + 1 <= r[bix]){
								pfix[bix * cols + inx] = calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m],nums[bix * cols + l[bix] + inx * m + 1]) >= calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m + 1],num);
							}
				
							__syncthreads();
							if(inx + 1 < cols && pfix[bix * cols + inx + 1] != pfix[bix * cols + inx]){
								r[bix] = min(l[bix] + (inx + 1) * m + 1,r[bix]);
								l[bix] = l[bix] + inx * m;
							}
							__syncthreads();
									
						}
						if(inx == 0){
							nums[bix * cols + r[bix]] = num;     
						}
			
					}	
				}
				
			}
			if(inx == 0)
				r[bix]++;
		}
		
	}
	
}
// 将结果初始化为无限大
/// @brief 
/// @param res		结果数组 
/// @param size 	数组长度
/// @return 
__global__ void res_init(double * res,int size){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	for(;inx < size;inx += threadSize){
		res[inx] = 9000 * 9000;
	}
}
//根据上行的邻近点使用抽样优化的逐一添加方法计算出本行邻近点
/// @brief 
/// @param nums		点数组 
/// @param rows 	行数
/// @param cols 	列数
/// @param n 		分块大小
/// @param r 		抽样分区终点
/// @param l 		抽样分区起始点
/// @param pfix 	前缀和数组
/// @return 
__global__ void step3(GPoint * nums,int rows,int cols,int n,int * r,int * l,int * pfix){
	int bix = blockIdx.x * n;
	int threadSize = blockDim.x;
	int inx = threadIdx.x;
	int blockSize = gridDim.x - 1;
	int end;
	int p1,p2,m;
	int size;
	GPoint num;
	for(;bix < rows;bix += blockSize * n){
		end = min(bix + n,rows);
		for(bix++;bix < end;bix++){
    	__syncthreads();
		p1 = 0;
		p2 = r[bix];
      	size = r[bix - 1];
		__syncthreads();
      	if(inx == 0)
        r[bix] = -1;
     	 __syncthreads();
 			while(p1 < size && p2 < cols){	
        	__syncthreads();
				if(nums[bix * cols + p2].x > nums[bix * cols - cols + p1].x){
					num.y = nums[bix * cols - cols + p1].y + 1;
					num.x = nums[bix * cols - cols + p1].x;	
					p1++;
				}
				else{
					num.y = nums[bix * cols + p2].y;
					num.x = nums[bix * cols + p2].x;
					if(nums[bix * cols + p2].x == nums[bix * cols - cols + p1].x)
						p1++;
					p2++;
				
        		}
        
				if(r[bix] < 1 || calculateVerticalBisectorX(nums[bix * cols + r[bix] - 1],nums[bix * cols + r[bix]]) < calculateVerticalBisectorX(nums[bix * cols +  r[bix]],num)){
					__syncthreads();
					if(inx == 0){
						r[bix]++;
						nums[bix * cols + r[bix]] = num;
					} 
				}
				else if(calculateVerticalBisectorX(nums[bix * cols],nums[bix * cols + 1]) >= calculateVerticalBisectorX(nums[bix * cols + 1],num)){
					__syncthreads();
					if(inx == 0){
						r[bix] = 1;
						nums[bix * cols + 1] = num;
					}
				}else{
					if(inx == 0)
						l[bix] = 0;
					__syncthreads();
					while(r[bix] - l[bix] > 2){
						m = (int)ceil((double)(r[bix] - l[bix]) / threadSize);
						pfix[bix * cols + min(inx,cols - 1)] = 1;
						if(inx == 0){
							pfix[bix * cols + min(cols - 1,threadSize)] = 1;
						}
						__syncthreads();
						if(l[bix] + inx * m + 1 <= r[bix]){
							pfix[bix * cols + inx] = calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m],nums[bix * cols + l[bix] + inx * m + 1]) >= calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m + 1],num);
						}
						__syncthreads();
						if(inx + 1 < cols && pfix[bix * cols + inx + 1] != pfix[bix * cols + inx]){
							r[bix] = min(l[bix] + (inx + 1) * m + 1,r[bix]);
							l[bix] = l[bix] + inx * m;
						}
						__syncthreads();
					}
					if(inx == 0){
						nums[bix * cols + r[bix]] = num;
					}
				}
			}
			while(p1 < size){
				num.y = nums[bix * cols - cols + p1].y + 1;
				num.x = nums[bix * cols - cols + p1].x;
        		if(r[bix] < 1 || calculateVerticalBisectorX(nums[bix * cols + r[bix] - 1],nums[bix * cols + r[bix]]) < calculateVerticalBisectorX(nums[bix * cols +  r[bix]],num)){
					__syncthreads();
					if(inx == 0){
						r[bix]++;
						nums[bix * cols + r[bix]] = num;
					} 
				}
				else if(calculateVerticalBisectorX(nums[bix * cols],nums[bix * cols + 1]) >= calculateVerticalBisectorX(nums[bix * cols + 1],num)){
					__syncthreads();
					if(inx == 0){
						r[bix] = 1;
						nums[bix * cols + 1] = num;
					}
				}else{
					if(inx == 0)
						l[bix] = 0;
					__syncthreads();
					while(r[bix] - l[bix] > 2){
						m = (int)ceil((double)(r[bix] - l[bix]) / threadSize);
						pfix[bix * cols + min(inx,cols - 1)] = 1;
						if(inx == 0){
							pfix[bix * cols + min(cols - 1,threadSize)] = 1;
						}
						__syncthreads();
						if(l[bix] + inx * m + 1 <= r[bix]){
							pfix[bix * cols + inx] = calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m],nums[bix * cols + l[bix] + inx * m + 1]) >= calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m + 1],num);
						}
						__syncthreads();
						if(inx + 1 < cols && pfix[bix * cols + inx + 1] != pfix[bix * cols + inx]){
							r[bix] = min(l[bix] + (inx + 1) * m + 1,r[bix]);
							l[bix] = l[bix] + inx * m;
						}
						__syncthreads();
					}
					if(inx == 0){
						nums[bix * cols + r[bix]] = num;
					}
				}
				p1++;

			}

			while(p2 < cols){
				num.x = nums[bix * cols + p2].x;
				num.y = nums[bix * cols + p2].y;
        		if(r[bix] < 1 || calculateVerticalBisectorX(nums[bix * cols + r[bix] - 1],nums[bix * cols + r[bix]]) < calculateVerticalBisectorX(nums[bix * cols +  r[bix]],num)){
					__syncthreads();
					if(inx == 0){
						r[bix]++;
						nums[bix * cols + r[bix]] = num;
					} 
				}
				else if(calculateVerticalBisectorX(nums[bix * cols],nums[bix * cols + 1]) >= calculateVerticalBisectorX(nums[bix * cols + 1],num)){
					__syncthreads();
					if(inx == 0){
						r[bix] = 1;
						nums[bix * cols + 1] = num;
					}
				}else{
					if(inx == 0)
						l[bix] = 0;
					__syncthreads();
					while(r[bix] - l[bix] > 2){
						m = (int)ceil((double)(r[bix] - l[bix]) / threadSize);
						pfix[bix * cols + min(inx,cols - 1)] = 1;
						if(inx == 0){
							pfix[bix * cols + min(cols - 1,threadSize)] = 1;
						}
						__syncthreads();
						if(l[bix] + inx * m + 1 <= r[bix]){
							pfix[bix * cols + inx] = calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m],nums[bix * cols + l[bix] + inx * m + 1]) >= calculateVerticalBisectorX(nums[bix * cols + l[bix] + inx * m + 1],num);
						}
						__syncthreads();
						if(inx + 1 < cols && pfix[bix * cols + inx + 1] != pfix[bix * cols + inx]){
							r[bix] = min(l[bix] + (inx + 1) * m + 1,r[bix]);
							l[bix] = l[bix] + inx * m;
						}
						__syncthreads();
					}
					if(inx == 0){
						nums[bix * cols + r[bix]] = num;
					}
				}
				p2++;
			}
			
			if(inx == 0)
				r[bix]++;
		}
     
	}
}
//	计算出每个像素点与其所属邻近点的距离
/// @brief 
/// @param nums		点数组 
/// @param length 	点数组长度数组
/// @param rows 	行数
/// @param cols 	列数
/// @param res 		结果数组
/// @return 
__global__ void step41(GPoint *nums,int *length,int rows,int cols,double *res){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int bolck = blockDim.x * gridDim.x;
	int left;
	int right;
	for(;i < rows; i+= bolck){
		left = 0;
		for(int j = 0;j < length[i] - 1;j++){
			right = min(cols - 1,(int)floor(calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + j + 1])));
			while(left <= right){
				res[i * cols + left] = (nums[i * cols + j].y) * (nums[i * cols + j].y) + (left - nums[i * cols + j].x) * (left - nums[i * cols + j].x);
				left++;	
			}
		}
		right = cols - 1;
		while(left <= right){
				res[i * cols + left] = (nums[i * cols + length[i] - 1].y) * (nums[i * cols + length[i] - 1].y) + (left - nums[i * cols + length[i] - 1].x) * (left - nums[i * cols + length[i] - 1].x);
			left++;
		}	
	}
}
//	计算出每个像素点与其所属邻近点的距离
/// @brief 
/// @param nums		点数组 
/// @param length 	点数组长度数组
/// @param rows 	行数
/// @param cols 	列数
/// @param res 		结果数组
/// @return 
__global__ void step42(GPoint *nums,int *length,int rows,int cols,double *res){
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int bolck = blockDim.x * gridDim.x;
	for(;i < rows; i+= bolck){
		int left = 0;
		int right;
		for(int j = 0;j < length[i] - 1;j++){
			right = min(cols - 1,(int)floor(calculateVerticalBisectorX(nums[i * cols + j],nums[i * cols + j + 1])));
			while(left <= right){
				res[i * cols + left] = min(res[i * cols + left],(double)(nums[i * cols + j].y) * (nums[i * cols + j].y) + (left - nums[i * cols + j].x) * (left - nums[i * cols + j].x));
				left++;	
			}
		}
		right = cols - 1;
		while(left <= right){
				res[i * cols + left] = min(res[i * cols + left],(double)(nums[i * cols + length[i] - 1].y) * (nums[i * cols + length[i] - 1].y) + (left - nums[i * cols + length[i] - 1].x) * (left - nums[i * cols + length[i] - 1].x));
			left++;
		}	
	}
	
}

// 结果验证算法
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

// 结果验证算法
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
			left[j] = 
			left[j - 1];
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


GpuMat tranformation(GpuMat &g_input, Stream &stream,vector<vector<bool>>& flag,int n,int threadSize,int ans_flag) {  //&的意思是实参可以被改变

    //将图像读入GPU
    GpuMat g_output;
    // 在gpu里创建一块连续的空间。
    g_output = createContinuous(g_input.size(), g_input.type());
    g_input.copyTo(g_output, stream);
	uint64_t t1, t2;
	cudaEvent_t start_event, stop_event;
	float time_kernel; 
    
    int rows = g_input.rows;  //图像的行数
	int cols = g_input.cols;  //图像的列数

	int m = min(512,max(1,rows / n));
	GPoint * nums; //点数组
	int * pfix; //前缀和数组
	int * nums_size; //点数组长度记录数组
    double * ans; //结果记录数组
	double * cpu_ans = (double*)malloc(sizeof(double) * rows * cols);
    (cudaMalloc((void**)&nums,sizeof(GPoint) * rows * cols));
    (cudaMalloc((void**)&nums_size,sizeof(int) * rows));
    (cudaMalloc((void**)&pfix,sizeof(int) * rows * cols));
	(cudaMalloc((void**)&ans,sizeof(double) * rows * cols));

   	int * l;
	(cudaMalloc((void**)&l,sizeof(int) * rows));
	//将结果初始化为无限大
	res_init<<<16,512>>>(ans,rows * cols);
	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//步骤一：计算每个像素点上方最近黑点的距离
	step1<<<m,512>>>(g_output.data,nums,rows,cols);

	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step1 time %fus\n", time_kernel * 1000);
    

	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//步骤二：计算每个块的第一行的邻近点，将剩余行的黑点压缩至数组末尾
	step2<<<16,512>>>(nums,rows,cols,n,nums_size);
  
	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step2 time %fus\n", time_kernel * 1000);	
    
	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//步骤三：根据上行的邻近点使用抽样优化的逐一添加方法计算出本行邻近点
 	step3<<<min(m,512 * 8),threadSize>>>(nums,rows,cols,n,nums_size,l,pfix);

	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step3 time %fus\n", time_kernel * 1000);
   
	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//步骤四：计算出每个像素点与其所属邻近点的距离
	step41<<<16,512>>>(nums,nums_size,rows,cols,ans);

	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step4 time %fus\n", time_kernel * 1000);

	// 图片矩阵转置，将图片倒置
	tan1<<<m,512>>>(g_output.data,rows,cols);


	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//步骤一：计算每个像素点上方最近黑点的距离
	step1<<<16,512>>>(g_output.data,nums,rows,cols);

	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step1 time %fus\n", time_kernel * 1000);
    

	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);

	//步骤二：计算每个块的第一行的邻近点，将剩余行的黑点压缩至数组末尾
	step2<<<16,512>>>(nums,rows,cols,n,nums_size);
	
	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step2 time %fus\n", time_kernel * 1000);							

	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);

	//步骤三：根据上行的邻近点使用抽样优化的逐一添加方法计算出本行邻近点
	step3<<<min(m,512 * 8),threadSize>>>(nums,rows,cols,n,nums_size,l,pfix);
  
  	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step3 time %fus\n", time_kernel * 1000);
	// nums转置
    tan2<<<m,512>>>(nums,rows,cols);
	// nums_size转置
	tan3<<<m,512>>>(nums_size,rows);
	
	cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//步骤四：计算出每个像素点与其所属邻近点的距离
	step42<<<16,512>>>(nums,nums_size,rows,cols,ans);

	cudaDeviceSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("step4 time %fus\n", time_kernel * 1000);

	cudaMemcpy(cpu_ans,ans,sizeof(double) * rows * cols,cudaMemcpyDeviceToHost);
	// 结果检验
	if(ans_flag){
		double *fun = fun4(flag);
		int cnt = 0;
		for(int i = 0; i < rows * cols; i++){
			if(fun[i] != sqrt(cpu_ans[i])){
				//printf("%d  fun = %lf   gpu = %lf\n",i,fun[i],sqrt(cpu_ans[i]));
				cnt++;
			}
		}
		printf("cnt = %d\n",cnt);
	}
 
    stream.waitForCompletion();
    return g_input;
}


int main(int argc, char **argv) {
	
	if (argc != 5) {
		printf("Usage: ./edm <图片路径> <分块大小> <stpe3抽样优化线程组线程分配(每组最多可分配512个线程)> <是否需要检验结果：1(是) 0(否))>\n");
		return -1;
	}
	// imread 是opencv里一个把图片读进矩阵的函数  mat矩阵
	Mat input = imread(argv[1], -1);
	vector<vector<bool>> flag(input.rows,vector<bool>(input.cols));
	for(int i = 0; i < input.cols*input.rows; i++){
		flag[i / input.cols][i % input.cols] = input.at<uchar>(i / input.cols,i % input.cols) == 0;
	}
	// 结果图片矩阵
	Mat output;

	// 时间变量
	uint64_t t1, t2;
	// opencv中封装好的操控CUDA的函数
	Stream stream;
	// GPU中的图片矩阵矩阵
	GpuMat g_input;
	GpuMat g_output;
	// 上传图片矩阵到GPU中
	g_input.upload(input, stream);
	stream.waitForCompletion();//协同函数
	
	t1 = cci::common::event::timestampInUS(); //计时开始
	
	g_output = tranformation(g_input, stream,flag,atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));//tranformation函数在这里调用
	
	stream.waitForCompletion();
	t2 = cci::common::event::timestampInUS(); //计时结束


	cout << "total time :" << t2-t1 << "us" << endl;
	
	// 从GPU下载图片矩阵到CPU
	// g_output.download(output);

	// imwrite是opencv里一个把图片矩阵写到文件中函数
	// imwrite("result/result.png", output);
	// g_input.release();
	// g_output.release();
	
	return 0;
}



