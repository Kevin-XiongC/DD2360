all:
	nvcc -arch=sm_61 perceptron_gpu.cu -o perceptron_gpu.out
	./perceptron_gpu.out

run:
	./perceptron_gpu.out

cpu:
	g++ perceptron_cpu.cpp -o perceptron_cpu.out --std=c++11
	./perceptron_cpu.out

plot_cpu:
	python plots/plot.py ./plots/perceptron_cpu.txt ./plots/perceptron_cpu.png

plot_gpu:
	python plots/plot.py ./plots/perceptron_gpu.txt ./plots/perceptron_gpu.png
