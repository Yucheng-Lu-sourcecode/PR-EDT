#include<iostream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>

using namespace std;

struct Point{
	int x,y;
	Point(int _y,int _x){
		y = _y;
		x = _x;
	}
	Point(){
	}
};

Point* data1;
Point* data2;

double* fun1(bool* flag,int cols,int rows){
	Point* nums1 = (Point*)malloc(sizeof(Point) * cols * rows);
	Point* nums2 = (Point*)malloc(sizeof(Point) * cols * rows);
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
				data1[y * cols + x].y = nums1[j].y;
				data1[y * cols + x].x = nums1[j].x;
			}
		}
			
	}
	free(nums1);
	free(nums2);
	return res;
}

int f1(Point a,Point b){
	double d1 = (double)(a.y + b.y) / 2 * (double)(a.y - b.y) / (double)(a.x - b.x) + (double)(a.x + b.x) / 2;
	double d2 = double(int(d1));
	return d1 == d2 ? int(d1) : (int(d1) + 1);
}

int f2(Point a,Point b){
	return int((double)(a.y + b.y) / 2 * (double)(a.y - b.y) / (double)(a.x - b.x) + (double)(a.x + b.x) / 2);
}

double f3(Point a,Point b){
	return (double)(a.y + b.y) / 2 * (double)(a.y - b.y) / (double)(a.x - b.x) + (double)(a.x + b.x) / 2;
}

void fun3(Point* nums,int * size){
	int p = 1;
	int len = *size;
	for(int i = 2;i < len;i++){
		while(p > 0 && f3(nums[p - 1],nums[p]) >= f3(nums[p],nums[i])){
			p--;
		}
		nums[++p] = nums[i];
	}
	*size = p + 1;
	
}
void fun123(Point* nums,int * size){
	int len = *size;
	vector<Point> arr(len);
	for(int i = 0;i < len;i++){
		arr[i] = nums[i];
	}
	vector<bool> flag(len,true);
	int cnt = 0;
	int arr_size;
	do{
		cnt++;
		arr_size = len;
		for(int i = 0;i < arr_size;i++)
			flag[i] = true;
	for(int i = 0;i < arr_size;i++){
		for(int j = i - 2;j >= 0;j--){
			if(f3(arr[j],arr[j + 1]) < f3(arr[j + 1],arr[i]))
				break;
			flag[j + 1] = false;
		}
	} 
	len = 0;
	for(int i = 0;i < arr_size;i++){
		if(flag[i])
			arr[len++] = arr[i];
	}
	}while(arr_size != len);
	printf("cnt = %d\n",cnt);
	int p = 1;
	len = *size;
	for(int i = 2;i < len;i++){
		while(p > 0 && f3(nums[p - 1],nums[p]) >= f3(nums[p],nums[i])){
			p--;
		}
		nums[++p] = nums[i];
	}
	*size = p + 1;
		printf("len = %d  size is erreo arr = %d;nums = %d\n",len,arr_size,*size);
	for(int i = 0;i < arr_size;i++){
		if(arr[i].x != nums[i].x && arr[i].y != nums[i].y)
			printf("%d  is erreo arr(%d,%d)  nums(%d,%d)\n",i,arr[i].x,arr[i].y,nums[i].x,nums[i].y);
	}
	
}

Point* me(Point * a,Point* b,int a_size,int b_size,int * returnSize){
	int left = 1;
	int right = a_size - 1;
	int mid = 0;
	bool flag = false;
	double x;
	while(left <= right){
		flag = false;
		mid = (left + right) >> 1;
		x = f3(a[mid],a[mid - 1]); 
		for(int i = 0;i < b_size;i++){
			if(x >= f3(a[mid],b[i])){
				flag = true;
				right = mid - 1;
				break;
			}
		}
		if(!flag){
			left = mid + 1;
		}
		
	}
	int p = flag ? mid - 1 : mid;
		
	left = 0;
	right = b_size - 2;
	flag = false;
	mid = 0;
	while(left <= right){
		flag = false;
		mid = (left + right) >> 1;
		x = f3(b[mid],b[mid + 1]);
		for(int i = a_size - 1;i >= 0;i--){
			if(x <= f3(b[mid],a[i])){
				flag = true;
				left = mid + 1;
				break;
			}
		}
		if(!flag)
			right = mid - 1;
	}
	int q = flag ? mid + 1 : mid;
	Point* ans = (Point*)malloc(sizeof(Point) * (p + 1 + b_size - q));
	*returnSize = 0;
	for(int i = 0;i <= p;i++)
		ans[(*returnSize)++] = a[i];
	for(int i = q;i < b_size;i++)
		ans[(*returnSize)++]= b[i]; 
	return ans;
}

Point* me1(Point * a,Point* b,int a_size,int b_size,int * returnSize){
	int left = 1;
	int right = a_size - 1;
	int mid = 0;
	bool flag = false;
	double x;
	int left1,right1,mid1;
	bool flag1;
	while(left <= right){
		flag = false;
		mid = (left + right) >> 1;
		x = f3(a[mid],a[mid - 1]); 
		left1 = 0,right1 = b_size - 2;
		mid1 = 0;
		flag1 = false;
		while(left1 <= right1){
			flag1 = false;
			mid1 = (left1 + right1) >> 1;
			(flag1 = (f3(a[mid],b[mid1]) >= f3(b[mid1],b[mid1 + 1]))) ? left1 = mid1 + 1 : right1 = mid1 - 1;
		}
		if(flag1)
			mid1++;
		(flag = (x >= f3(a[mid],b[mid1]))) ? right = mid - 1 : left = mid + 1;
		
	}
	int p = flag ? mid - 1 : mid;
		
	left = 0;
	right = b_size - 2;
	flag = false;
	mid = 0;
	while(left <= right){
		flag = false;
		mid = (left + right) >> 1;
		x = f3(b[mid],b[mid + 1]);
		left1 = 1,right1 = a_size - 1;
		mid1 = 0;
		flag1 = false;
		while (left1 <= right1){
			flag1 = false;
			mid1 = (left1 + right1) >> 1;
			(flag1 = (f3(a[mid1],b[mid]) <= f3(a[mid1 - 1],a[mid1]))) ? right1 = mid1 - 1 : left1 = mid1 + 1;
		}
		if(flag1)
			mid1--;
		(flag = (x <= f3(a[mid1],b[mid]))) ? left = mid + 1 : right = mid - 1;
		
	}
	int q = flag ? mid + 1 : mid;
	Point* ans = (Point*)malloc(sizeof(Point) * (p + 1 + b_size - q));
	*returnSize = 0;
	for(int i = 0;i <= p;i++)
		ans[(*returnSize)++] = a[i];
	for(int i = q;i < b_size;i++)
		ans[(*returnSize)++]= b[i]; 
	return ans;
}

Point* fun4(Point* nums,int left,int right,int * returnSize){
	if(left < right){
		int a_size = 0;
		Point* a = fun4(nums,left,(left + right) >> 1,&a_size);
		int b_size = 0; 
		Point* b = fun4(nums,((left + right) >> 1) + 1,right,&b_size);
		Point* res = me(a,b,a_size,b_size,returnSize);
        free(a);
		free(b); 
		return res;
	}
	else{
		*returnSize = 1;
		Point* res = (Point*)malloc(sizeof(Point));
		res[0] = nums[left];
		return res;
	}
}

double* fun2(bool* flag,int cols,int rows){
	double * res = (double*)malloc(sizeof(double) * rows * cols);
	int * nums = (int*)malloc(sizeof(int) * rows * cols);
	
		
	time_t st;
	time_t ed;	
	st = clock();
	
	
	
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++){
			nums[i * cols + j] = 9999999;
		}
	}
	for(int j = 0;j < cols;j++)
		if(flag[j])
			nums[j] = 0;
	for(int i = 1;i < rows;i++){
		for(int j = 0;j < cols;j++){
			if(flag[i * cols + j])
				nums[i * cols + j] = 0;
			else 
				nums[i * cols + j] =  nums[(i - 1) * cols + j] + 1;
		}
	}
	for(int i = rows - 2;i >= 0;i--){
		for(int j = 0;j < cols;j++)
			nums[i * cols + j] = min(nums[i * cols + j],nums[(i + 1) * cols + j] + 1);	
	}
	
	ed = clock();
	cout << "step1: " << ed - st << "ms" << endl;
	
	Point * arr = (Point*)malloc(sizeof(Point) * rows * cols);
	int * arr_size = (int*)malloc(sizeof(int) * rows);
	
	st = clock();
	for(int i = 0;i < rows;i++)
		arr_size[i] = 0;
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++){
			if(nums[i * cols + j] < 9999999){
				arr[i * cols + arr_size[i]].y = nums[i * cols + j];
				arr[i * cols + arr_size[i]].x = j;
				arr_size[i]++;
			}
        }
				
		fun123(arr + i * cols,&arr_size[i]); 
	}
	return NULL;
	ed = clock();
	cout << "step2: " << ed - st << "ms" << endl;

	st = clock();
	int left,right;
	for(int i = 0;i < rows;i++){
		left = 0;
		for(int j = 0;j < arr_size[i] - 1;j++){
			right = min((int)cols - 1,(int)floor(f3(arr[i * cols + j],arr[i * cols + j + 1])));
			while(left <= right){
					res[i * cols + left] = sqrt((arr[i * cols + j].y) * (arr[i * cols + j].y) + (left - arr[i * cols + j].x) * (left - arr[i * cols + j].x));
				left++;
				
			}
		}
		right = cols - 1;
		while(left <= right){
					res[i * cols + left] = sqrt((arr[i * cols + arr_size[i] - 1].y) * (arr[i * cols + arr_size[i] - 1].y) + (left - arr[i * cols + arr_size[i] - 1].x) * (left - arr[i * cols + arr_size[i] - 1].x));
			left++;
		}	
	}
		
	ed = clock();
	cout << "step3: " << ed - st << "ms" << endl;
	free(nums);
	free(arr);
	free(arr_size);
	return res;
}

int fun12(Point* nums,int size,Point num){
	while(size >= 2 && f3(nums[size - 2],nums[size - 1]) >= f3(nums[size - 1],num)){
		size--;
	}
	nums[size] = num;
	return size + 1;
}

double* fun11(bool* flag,int cols,int rows){
	time_t st;
	time_t ed;
	
	double * res = (double*)malloc(sizeof(double) * rows * cols);

	Point * arr = (Point*)malloc(sizeof(Point) * rows * cols);
	int * arr_size = (int*)malloc(sizeof(int) * rows);
	st = clock();
	for(int i = 0;i < rows;i++)
		arr_size[i] = 0;
	for(int j = 0;j < cols;j++){
		if(flag[j]){
			arr[arr_size[0]].y = 0;
			arr[arr_size[0]].x = j;
			arr_size[0]++;
		}
    }

    int p1,p2,p;
	for(int i = 1;i < rows;i++){
		p2 = cols;
		for(int j = cols - 1;j >= 0;j--){
			if(flag[i * cols + j]){
				p2--;
				arr[i * cols + p2].y = 0;
				arr[i * cols + p2].x = j;
			}
        }
		p1 = 0;
		p = 0;	
		while(p1 < arr_size[i - 1] && p2 < cols){
			if(arr[i * cols + p2].x > arr[i * cols - cols + p1].x){
				p = fun12(&arr[i * cols],p,Point(arr[i * cols - cols + p1].y + 1,arr[i * cols - cols + p1].x));	
				p1++;
			}
			else{
				p = fun12(&arr[i * cols],p,arr[i * cols + p2]);
				if(arr[i * cols + p2].x == arr[i * cols - cols + p1].x)
					p1++;
				p2++;
			}
		}
		while(p1 < arr_size[i - 1]){
			p = fun12(&arr[i * cols],p,Point(arr[i * cols - cols + p1].y + 1,arr[i * cols - cols + p1].x));
			p1++;
		} 
		while(p2 < cols){
			p = fun12(&arr[i * cols],p,arr[i * cols + p2]);
			p2++;
		}
		arr_size[i] = p;
	}	
	
	int left,right;
	for(int i = 0;i < rows;i++){
		left = 0;
		for(int j = 0;j < arr_size[i] - 1;j++){
			right = min((int)cols - 1,(int)floor(f3(arr[i * cols + j],arr[i * cols + j + 1])));
			while(left <= right){
					res[i * cols + left] = sqrt((arr[i * cols + j].y) * (arr[i * cols + j].y) + (left - arr[i * cols + j].x) * (left - arr[i * cols + j].x));
				left++;
				
			}
		}
		right = cols - 1;
		while(left <= right){
					res[i * cols + left] = sqrt((arr[i * cols + arr_size[i] - 1].y) * (arr[i * cols + arr_size[i] - 1].y) + (left - arr[i * cols + arr_size[i] - 1].x) * (left - arr[i * cols + arr_size[i] - 1].x));
			left++;
		}	
	}
		
	for(int i = 0;i < rows;i++)
		arr_size[i] = 0;
	for(int j = 0;j < cols;j++){
		if(flag[rows * cols - cols + j]){
			arr[rows * cols - cols + arr_size[rows - 1]].y = 0;
			arr[rows * cols - cols + arr_size[rows - 1]].x = j;
			arr_size[rows - 1]++;
		}
    }
	for(int i = rows - 2;i >= 0;i--){
		p2 = cols;
		for(int j = cols - 1;j >= 0;j--){
			if(flag[i * cols + j]){
				p2--;
				arr[i * cols + p2].y = 0;
				arr[i * cols + p2].x = j;
			}
        }
		p1 = 0;
		p = 0;	
		while(p1 < arr_size[i + 1] && p2 < cols){
			if(arr[i * cols + p2].x > arr[i * cols + cols + p1].x){
				p = fun12(&arr[i * cols],p,Point(arr[i * cols + cols + p1].y + 1,arr[i * cols + cols + p1].x));	
				p1++;
			}
			else{
				p = fun12(&arr[i * cols],p,arr[i * cols + p2]);
				if(arr[i * cols + p2].x == arr[i * cols + cols + p1].x)
					p1++;
				p2++;
			}
		}
		while(p1 < arr_size[i + 1]){
			p = fun12(&arr[i * cols],p,Point(arr[i * cols + cols + p1].y + 1,arr[i * cols + cols + p1].x));
			p1++;
		}
		 
		while(p2 < cols){
			p = fun12(&arr[i * cols],p,arr[i * cols + p2]);
			p2++;
		}
		arr_size[i] = p;
	}	
	
	for(int i = 0;i < rows;i++){
		left = 0;
		for(int j = 0;j < arr_size[i] - 1;j++){
			right = min((int)cols - 1,(int)floor(f3(arr[i * cols + j],arr[i * cols + j + 1])));
			while(left <= right){
					res[i * cols + left] = min(res[i * cols + left],sqrt((arr[i * cols + j].y) * (arr[i * cols + j].y) + (left - arr[i * cols + j].x) * (left - arr[i * cols + j].x)));
				left++;
				
			}
		}
		right = cols - 1;
		while(left <= right){
					res[i * cols + left] = min(res[i * cols + left],sqrt((arr[i * cols + arr_size[i] - 1].y) * (arr[i * cols + arr_size[i] - 1].y) + (left - arr[i * cols + arr_size[i] - 1].x) * (left - arr[i * cols + arr_size[i] - 1].x)));
			left++;
		}	
	}	
	free(arr);
	free(arr_size);
	return res;
}
int main(){
	
	time_t st;
	time_t ed;
	time_t run_time;
	vector<int> rows({8192,4096,2048,1024,512});
	vector<int> cols({8192,4096,2048,1024,512});
	vector<int> r(99);
	for(int i = 0;i < 99;i++){
		r[i] = i + 1;
	}
	vector<int> n({4,7,6,7,4});
  	srand(5);
	uint64_t t1, t2;
	bool * flag;
	int m = 10;
	for(int i = 0;i < rows.size();i++){
		flag = (bool*)malloc(rows[i] * cols[i] * sizeof(bool));
		for(int j = 0;j < r.size();j++){
			
			for(int k = 0; k < rows[i]; k++){
				for(int l = 0;l < cols[i];l++)
					flag[k * cols[i] + l] = (rand() % 100) < r[j];

			}
			run_time = 0;
			for(int k = 0;k < m;k++){
				st = clock();
				double * res1 = fun11(flag,rows[i],cols[i]);
				ed = clock();
				run_time += (ed - st);
				free(res1);
			}
			cout << run_time / m << endl;
		}
		free(flag);
	}
	
	return 0;
} 