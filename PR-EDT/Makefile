compile:
	nvcc -G -g -o edm edm.cu -rdc=true -L"/usr/local/lib" -I"/usr/local/include/opencv4"  -lopencv_core -lopencv_imgcodecs -std=c++11
run:
	./PR-EDT ./image/L-1024.png
	#./PR-EDT ./image/L-2048.png
	#./PR-EDT ./image/L-4096.png
	#./PR-EDT ./image/L-8192.png
gdb:
	cuda-gdb ./PR-EDT