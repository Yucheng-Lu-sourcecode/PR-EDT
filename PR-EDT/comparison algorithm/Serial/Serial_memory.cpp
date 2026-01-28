#include<iostream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<fstream>
#include<string>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>

using namespace std;

// 解决命名冲突：自定义点类型重命名为MyPoint
struct MyPoint{
        int x,y;
        MyPoint(int _y,int _x){
                y = _y;
                x = _x;
        }
        MyPoint(){}
};

MyPoint* data1;
MyPoint* data2;

// ========== 内存统计结构：支持单张图片独立重置和峰值+总量统计 ==========
struct MemoryStats {
    size_t current_heap;      // 当前已分配堆内存（用于计算峰值）
    size_t peak_heap;         // 单张图片处理期间的峰值堆内存
    size_t total_allocated;   // 新增：累计分配总量（所有 malloc 的 size 之和，不减 free）
} mem_stats;

// 初始化/重置内存统计（处理每张图片前调用）
void reset_memory_stats() {
    mem_stats.current_heap = 0;
    mem_stats.peak_heap = 0;
    mem_stats.total_allocated = 0;  // ← 新增重置
}

// 自定义malloc：更新当前内存、峰值、累计总量
void* my_malloc(size_t size) {
    if (size == 0) return NULL;
    void* ptr = malloc(size);
    if (ptr != NULL) {
        mem_stats.current_heap += size;
        mem_stats.total_allocated += size;  // ← 关键新增：只增不减
        if (mem_stats.current_heap > mem_stats.peak_heap) {
            mem_stats.peak_heap = mem_stats.current_heap;
        }
    }
    return ptr;
}

// 自定义free：仅更新当前内存，不修改峰值或总量
void my_free(void* ptr, size_t size) {
    if (ptr == NULL || size == 0) return;
    free(ptr);
    mem_stats.current_heap -= size;
    // 防止因重复释放导致下溢（简单保护）
    if (mem_stats.current_heap > (1ULL << 40)) { // 超过 1TB 视为异常
        mem_stats.current_heap = 0;
    }
}

// 打印单张图片的峰值内存和累计分配总量
void print_single_image_peak_memory(const string& image_name) {
    double peak_mb = (double)mem_stats.peak_heap / (1024 * 1024);
    double total_mb = (double)mem_stats.total_allocated / (1024 * 1024);

    cout << "=====================================" << endl;
    cout << "📊 图片 [" << image_name << "] 内存统计" << endl;
    cout << "=====================================" << endl;
    cout << "峰值内存（Peak）：" << fixed << peak_mb << " MB" << endl;
    cout << "累计分配（Total Allocated）：" << total_mb << " MB" << endl;
    cout << "=====================================" << endl << endl;
}
// ==========================================

// 暴力算法（仅保留内存分配释放，无额外打印）
double* fun1(bool* flag,int cols,int rows){
    MyPoint* nums1 = (MyPoint*)my_malloc(sizeof(MyPoint) * cols * rows);
    MyPoint* nums2 = (MyPoint*)my_malloc(sizeof(MyPoint) * cols * rows);
    int nums1_size = 0, nums2_size = 0; 
    double* res = (double*)my_malloc(sizeof(double) * cols * rows);

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
                y = nums2[i].y; x = nums2[i].x;
                for(int j = 0;j < nums1_size;j++){
                        if(res[y * cols + x] > sqrt(pow(y - nums1[j].y,2) + pow(x - nums1[j].x,2))){
                                res[y * cols + x] = sqrt(pow(y - nums1[j].y,2) + pow(x - nums1[j].x,2));
                                if (data1) {
                                        data1[y * cols + x].y = nums1[j].y;
                                        data1[y * cols + x].x = nums1[j].x;
                                }
                        }
                }
        }

        my_free(nums1, sizeof(MyPoint) * cols * rows);
        my_free(nums2, sizeof(MyPoint) * cols * rows);
        return res;
}

// 辅助函数：三点计算相关（无内存操作，仅算法逻辑）
int f1(MyPoint a,MyPoint b){
        double d1 = (double)(a.y + b.y)/2 * (double)(a.y - b.y)/(double)(a.x - b.x) + (double)(a.x + b.x)/2;
        double d2 = double(int(d1));
        return d1 == d2 ? int(d1) : (int(d1) + 1);
}
int f2(MyPoint a,MyPoint b){
        return int((double)(a.y + b.y)/2 * (double)(a.y - b.y)/(double)(a.x - b.x) + (double)(a.x + b.x)/2);
}
double f3(MyPoint a,MyPoint b){
        return (double)(a.y + b.y)/2 * (double)(a.y - b.y)/(double)(a.x - b.x) + (double)(a.x + b.x)/2;
}
void fun3(MyPoint* nums,int * size){
        int p = 1; int len = *size;
        for(int i = 2;i < len;i++){
                while(p > 0 && f3(nums[p-1],nums[p]) >= f3(nums[p],nums[i])) p--;
                nums[++p] = nums[i];
        }
        *size = p + 1;
}
void fun123(MyPoint* nums,int * size){
        int len = *size;
        vector<MyPoint> arr(len);
        for(int i = 0;i < len;i++) arr[i] = nums[i];
        vector<bool> flag(len,true);
        int cnt = 0, arr_size;
        do{
                cnt++; arr_size = len;
                for(int i = 0;i < arr_size;i++) flag[i] = true;
                for(int i = 0;i < arr_size;i++){
                        for(int j = i - 2;j >= 0;j--){
                                if(f3(arr[j],arr[j+1]) < f3(arr[j+1],arr[i])) break;
                                flag[j+1] = false;
                        }
                }
                len = 0;
                for(int i = 0;i < arr_size;i++) if(flag[i]) arr[len++] = arr[i];
        }while(arr_size != len);

        int p = 1; len = *size;
        for(int i = 2;i < len;i++){
                while(p > 0 && f3(nums[p-1],nums[p]) >= f3(nums[p],nums[i])) p--;
                nums[++p] = nums[i];
        }
        *size = p + 1;
}
MyPoint* me(MyPoint * a,MyPoint* b,int a_size,int b_size,int * returnSize){
        int left = 1, right = a_size -1, mid = 0;
        bool flag = false; double x;
        while(left <= right){
                flag = false; mid = (left + right) >> 1; x = f3(a[mid],a[mid-1]);
                for(int i = 0;i < b_size;i++){
                        if(x >= f3(a[mid],b[i])){ flag = true; right = mid -1; break; }
                }
                if(!flag) left = mid + 1;
        }
        int p = flag ? mid -1 : mid;
        left = 0; right = b_size -2; flag = false; mid = 0;
        while(left <= right){
                flag = false; mid = (left + right) >> 1; x = f3(b[mid],b[mid+1]);
                for(int i = a_size -1;i >= 0;i--){
                        if(x <= f3(b[mid],a[i])){ flag = true; left = mid +1; break; }
                }
                if(!flag) right = mid -1;
        }
        int q = flag ? mid +1 : mid;
        MyPoint* ans = (MyPoint*)my_malloc(sizeof(MyPoint) * (p +1 + b_size - q));
        *returnSize = 0;
        for(int i = 0;i <= p;i++) ans[(*returnSize)++] = a[i];
        for(int i = q;i < b_size;i++) ans[(*returnSize)++] = b[i];
        return ans;
}
MyPoint* me1(MyPoint * a,MyPoint* b,int a_size,int b_size,int * returnSize){
        int left =1, right =a_size-1, mid=0; bool flag=false; double x;
        int left1,right1,mid1; bool flag1;
        while(left <= right){
                flag=false; mid=(left+right)>>1; x=f3(a[mid],a[mid-1]);
                left1=0,right1=b_size-2; mid1=0; flag1=false;
                while(left1 <= right1){
                        flag1 = (f3(a[mid],b[mid1]) >= f3(b[mid1],b[mid1+1]));
                        flag1 ? left1=mid1+1 : right1=mid1-1;
                }
                if(flag1) mid1++;
                flag = (x >= f3(a[mid],b[mid1])) ? true : false;
                flag ? right=mid-1 : left=mid+1;
        }
        int p = flag ? mid-1 : mid;
        left=0; right=b_size-2; flag=false; mid=0;
        while(left <= right){
                flag=false; mid=(left+right)>>1; x=f3(b[mid],b[mid+1]);
                left1=1,right1=a_size-1; mid1=0; flag1=false;
                while(left1 <= right1){
                        flag1 = (f3(a[mid1],b[mid]) <= f3(a[mid1-1],a[mid1]));
                        flag1 ? right1=mid1-1 : left1=mid1+1;
                }
                if(flag1) mid1--;
                flag = (x <= f3(a[mid1],b[mid])) ? true : false;
                flag ? left=mid+1 : right=mid-1;
        }
        int q = flag ? mid+1 : mid;
        MyPoint* ans = (MyPoint*)my_malloc(sizeof(MyPoint)*(p+1 + b_size -q));
        *returnSize=0;
        for(int i=0;i<=p;i++) ans[(*returnSize)++] = a[i];
        for(int i=q;i<b_size;i++) ans[(*returnSize)++] = b[i];
        return ans;
}
MyPoint* fun4(MyPoint* nums,int left,int right,int * returnSize){
        if(left < right){
                int a_size=0, b_size=0;
                MyPoint* a = fun4(nums,left,(left+right)>>1,&a_size);
                MyPoint* b = fun4(nums,((left+right)>>1)+1,right,&b_size);
                MyPoint* res = me(a,b,a_size,b_size,returnSize);
                my_free(a, sizeof(MyPoint) * a_size);
                my_free(b, sizeof(MyPoint) * b_size);
                return res;
        }
        else{
                *returnSize =1;
                MyPoint* res = (MyPoint*)my_malloc(sizeof(MyPoint));
                res[0] = nums[left];
                return res;
        }
}
int fun12(MyPoint* nums,int size,MyPoint num){
        while(size >=2 && f3(nums[size-2],nums[size-1]) >= f3(nums[size-1],num)) size--;
        nums[size] = num;
        return size +1;
}

// 垂直平分线算法（fun2，无额外打印）
double* fun2(bool* flag,int cols,int rows){
    double * res = (double*)my_malloc(sizeof(double) * rows * cols);
    int * nums = (int*)my_malloc(sizeof(int) * rows * cols);
        time_t st = clock(), ed;

        for(int i = 0;i < rows;i++)
                for(int j = 0;j < cols;j++)
                        nums[i * cols + j] = 9999999;
        for(int j = 0;j < cols;j++)
                if(flag[j]) nums[j] = 0;
        for(int i = 1;i < rows;i++){
                for(int j = 0;j < cols;j++){
                        if(flag[i * cols + j]) nums[i * cols + j] = 0;
                        else nums[i * cols + j] =  nums[(i-1)*cols +j] +1;
                }
        }
        for(int i = rows-2;i >=0;i--)
                for(int j =0;j < cols;j++)
                        nums[i*cols+j] = min(nums[i*cols+j],nums[(i+1)*cols+j]+1);

        ed = clock();
        cout << "fun2 step1耗时：" << ed - st << "ms" << endl;

        MyPoint * arr = (MyPoint*)my_malloc(sizeof(MyPoint) * rows * cols);
        int * arr_size = (int*)my_malloc(sizeof(int) * rows);
        for(int i =0;i < rows;i++) arr_size[i] =0;
        for(int i =0;i < rows;i++){
                for(int j =0;j < cols;j++){
                        if(nums[i*cols+j] <9999999){
                                arr[i*cols+arr_size[i]].y = nums[i*cols+j];
                                arr[i*cols+arr_size[i]].x = j;
                                arr_size[i]++;
                        }
                }
                fun123(arr + i*cols,&arr_size[i]);
        }

        my_free(nums, sizeof(int) * rows * cols);
        my_free(arr, sizeof(MyPoint) * rows * cols);
        my_free(arr_size, sizeof(int) * rows);
        my_free(res, sizeof(double) * rows * cols);
        return NULL;
}

// 核心算法fun11（无额外内存打印，仅分配释放）
double* fun11(bool* flag,int cols,int rows){
    double * res = (double*)my_malloc(sizeof(double) * rows * cols);
    MyPoint * arr = (MyPoint*)my_malloc(sizeof(MyPoint) * rows * cols);
    int * arr_size = (int*)my_malloc(sizeof(int) * rows);
        int p1,p2,p;

        for(int i = 0;i < rows;i++) arr_size[i] = 0;
        for(int j = 0;j < cols;j++){
                if(flag[j]){
                        arr[arr_size[0]].y = 0;
                        arr[arr_size[0]].x = j;
                        arr_size[0]++;
                }
    }
        for(int i = 1;i < rows;i++){
                p2 = cols;
                for(int j = cols-1;j >=0;j--){
                        if(flag[i*cols+j]){
                                p2--;
                                arr[i*cols+p2].y =0;
                                arr[i*cols+p2].x =j;
                        }
        }
                p1 =0; p=0;
                while(p1 < arr_size[i-1] && p2 < cols){
                        if(arr[i*cols+p2].x > arr[i*cols-cols+p1].x){
                                p = fun12(&arr[i*cols],p,MyPoint(arr[i*cols-cols+p1].y+1,arr[i*cols-cols+p1].x));
                                p1++;
                        }
                        else{
                                p = fun12(&arr[i*cols],p,arr[i*cols+p2]);
                                if(arr[i*cols+p2].x == arr[i*cols-cols+p1].x) p1++;
                                p2++;
                        }
                }
                while(p1 < arr_size[i-1]){
                        p = fun12(&arr[i*cols],p,MyPoint(arr[i*cols-cols+p1].y+1,arr[i*cols-cols+p1].x));
                        p1++;
                } 
                while(p2 < cols){
                        p = fun12(&arr[i*cols],p,arr[i*cols+p2]);
                        p2++;
                }
                arr_size[i] = p;
        }

        int left,right;
        for(int i = 0;i < rows;i++){
                left =0;
                for(int j =0;j < arr_size[i]-1;j++){
                        right = min((int)cols-1,(int)floor(f3(arr[i*cols+j],arr[i*cols+j+1])));
                        while(left <= right){
                                res[i*cols+left] = sqrt(pow(arr[i*cols+j].y,2) + pow(left - arr[i*cols+j].x,2));
                                left++;
                        }
                }
                right = cols-1;
                while(left <= right){
                        res[i*cols+left] = sqrt(pow(arr[i*cols+arr_size[i]-1].y,2) + pow(left - arr[i*cols+arr_size[i]-1].x,2));
                        left++;
                }
        }

        for(int i = 0;i < rows;i++) arr_size[i] =0;
        for(int j =0;j < cols;j++){
                if(flag[rows*cols-cols+j]){
                        arr[rows*cols-cols+arr_size[rows-1]].y =0;
                        arr[rows*cols-cols+arr_size[rows-1]].x =j;
                        arr_size[rows-1]++;
                }
    }
        for(int i = rows-2;i >=0;i--){
                p2 = cols;
                for(int j = cols-1;j >=0;j--){
                        if(flag[i*cols+j]){
                                p2--;
                                arr[i*cols+p2].y =0;
                                arr[i*cols+p2].x =j;
                        }
        }
                p1 =0; p=0;
                while(p1 < arr_size[i+1] && p2 < cols){
                        if(arr[i*cols+p2].x > arr[i*cols+cols+p1].x){
                                p = fun12(&arr[i*cols],p,MyPoint(arr[i*cols+cols+p1].y+1,arr[i*cols+cols+p1].x));
                                p1++;
                        }
                        else{
                                p = fun12(&arr[i*cols],p,arr[i*cols+p2]);
                                if(arr[i*cols+p2].x == arr[i*cols+cols+p1].x) p1++;
                                p2++;
                        }
                }
                while(p1 < arr_size[i+1]){
                        p = fun12(&arr[i*cols],p,MyPoint(arr[i*cols+cols+p1].y+1,arr[i*cols+cols+p1].x));
                        p1++;
                }
                while(p2 < cols){
                        p = fun12(&arr[i*cols],p,arr[i*cols+p2]);
                        p2++;
                }
                arr_size[i] = p;
        }

        for(int i = 0;i < rows;i++){
                left =0;
                for(int j =0;j < arr_size[i]-1;j++){
                        right = min((int)cols-1,(int)floor(f3(arr[i*cols+j],arr[i*cols+j+1])));
                        while(left <= right){
                                res[i*cols+left] = min(res[i*cols+left],sqrt(pow(arr[i*cols+j].y,2) + pow(left - arr[i*cols+j].x,2)));
                                left++;
                        }
                }
                right = cols-1;
                while(left <= right){
                        res[i*cols+left] = min(res[i*cols+left],sqrt(pow(arr[i*cols+arr_size[i]-1].y,2) + pow(left - arr[i*cols+arr_size[i]-1].x,2)));
                        left++;
                }
        }

        // 释放中间内存，结果res由调用者释放
        my_free(arr, sizeof(MyPoint) * rows * cols);
        my_free(arr_size, sizeof(int) * rows);
        return res;
}

// 图片转二值化flag数组（cv::前缀访问OpenCV，无额外打印）
bool* image_to_flag(const string& image_path, int& out_rows, int& out_cols) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "❌ 无法读取图片：" << image_path << endl;
        out_rows = 0; out_cols = 0;
        return NULL;
    }
    out_rows = img.rows; out_cols = img.cols;
    bool* flag = (bool*)my_malloc(sizeof(bool) * out_rows * out_cols);
    if (!flag) {
        cerr << "❌ 内存分配失败：无法创建flag数组" << endl;
        out_rows = 0; out_cols = 0;
        return NULL;
    }
    // 二值化：0→true（黑点），非0→false（白点）
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            flag[i * out_cols + j] = (img.at<uchar>(i, j) == 0);
        }
    }
    cout << "✅ 读取图片：" << image_path << " | 尺寸：" << out_rows << "×" << out_cols << endl;
    return flag;
}

// ========== 主函数：每张图片独立统计峰值内存和累计分配量 ==========
int main(){
    srand(5);

    // 固定顺序：1024→2048→4096→8192
    vector<string> image_files = {
        "./image/L-1024.png",
        "./image/L-2048.png",
        "./image/L-4096.png",
        "./image/L-8192.png"
    };
    int m = 10;  // 每个图片运行算法次数（统计平均耗时）

    cout << "=====================================" << endl;
    cout << "开始处理 " << image_files.size() << " 张图片（固定顺序）" << endl;
    cout << "算法：fun11 | 每张运行次数：" << m << endl;
    cout << "内存指标：峰值内存 + 累计分配总量" << endl;
    cout << "=====================================" << endl << endl;

    // 遍历处理每张图片（每张独立统计峰值内存）
    for (size_t img_idx = 0; img_idx < image_files.size(); img_idx++) {
        // 步骤1：处理当前图片前，重置内存统计（清空上一张图片的内存数据）
        reset_memory_stats();
        string image_path = image_files[img_idx];
        string image_name = image_path.substr(image_path.find_last_of('/') + 1);
        int rows, cols;

        // 步骤2：读取图片并转换为flag数组
        bool* flag = image_to_flag(image_path, rows, cols);
        if (!flag || rows == 0 || cols == 0) {
            cout << "🔴 跳过该图片处理" << endl << endl;
            continue;
        }

        // 步骤3：运行m次fun11，统计平均耗时（同时累积当前图片的峰值内存）
        time_t total_time = 0;
        for (int k = 0; k < m; k++) {
            time_t st = clock();
            double* res = fun11(flag, cols, rows);
            time_t ed = clock();
            total_time += (ed - st);
            my_free(res, sizeof(double) * rows * cols); // 释放算法结果
        }

        // 步骤4：输出当前图片的耗时和内存统计
        double avg_time = (double)total_time / m;
        cout << "✅ 图片处理完成 | 平均耗时：" << avg_time << " ms" << endl;
        print_single_image_peak_memory(image_name); // 打印峰值 + 累计分配

        // 步骤5：释放当前图片的flag数组
        my_free(flag, sizeof(bool) * rows * cols);
    }

    // 程序结束
    cout << "✅ 所有图片处理完成！" << endl;
    return 0;
}
