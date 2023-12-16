from numba import cuda
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def gaussian_kernel_generator_cpu(besarkernel, delta):
    kernelRadius = besarkernel // 2
    result = np.zeros((besarkernel, besarkernel))

    pengali = 1 / (2 * np.pi * delta ** 2)

    for filterX in range(-kernelRadius, kernelRadius + 1):
        for filterY in range(-kernelRadius, kernelRadius + 1):
            result[filterY + kernelRadius, filterX + kernelRadius] = \
                math.exp(-(math.sqrt(filterY ** 2 + filterX ** 2) / (delta ** 2 * 2))) * pengali
    return result


@cuda.jit
def gaussian_kernel_cuda(kernel, besarKernel, delta):
    
    filterY, filterX = cuda.grid(2)

    kernelRadius = besarKernel // 2

    
    if filterX < besarKernel and filterY < besarKernel:
        x = filterX - kernelRadius
        y = filterY - kernelRadius
        pengali = 1 / (2 * np.pi * delta ** 2)
        kernel[filterY, filterX] = math.exp(-(math.sqrt(y ** 2 + x ** 2) / (2 * delta ** 2))) * pengali


def gaussian_kernel_generator(besarKernel, delta):
    
    result_device = cuda.device_array((besarKernel, besarKernel), dtype=np.float32)

    
    threadsperblock = (16, 16)

    
    blockspergrid_x = int(np.ceil(besarKernel / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(besarKernel / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    
    gaussian_kernel_cuda[blockspergrid, threadsperblock](result_device, besarKernel, delta)
    
    result_host = result_device.copy_to_host()
    return result_host


def gaussian_kernel_derivative_generator_cpu(besarKernel, delta):
    kernelRadius = besarKernel // 2
    resultX = np.zeros((besarKernel, besarKernel))
    resultY = np.zeros((besarKernel, besarKernel))

    pengali = -1 / (2 * np.pi * delta ** 4)

    for filterX in range(-kernelRadius, kernelRadius + 1):
        for filterY in range(-kernelRadius, kernelRadius + 1):
            resultX[filterY + kernelRadius, filterX + kernelRadius] = \
                np.exp(-(filterX ** 2 / (delta ** 2 * 2))) * pengali * filterX
            resultY[filterY + kernelRadius, filterX + kernelRadius] = \
                np.exp(-(filterY ** 2 / (delta ** 2 * 2))) * pengali * filterY
    return resultX, resultY


@cuda.jit
def gaussian_kernel_derivative_x_cuda(kernelX, besarKernel, delta):
    filterY, filterX = cuda.grid(2)
    kernelRadius = besarKernel // 2

    if filterX < besarKernel and filterY < besarKernel:
        x = filterX - kernelRadius
        pengali = -1 / (2 * np.pi * delta ** 4)
        kernelX[filterY, filterX] = math.exp(-(x ** 2 / (2 * delta ** 2))) * pengali * x


@cuda.jit
def gaussian_kernel_derivative_y_cuda(kernelY, besarKernel, delta):
    filterY, filterX = cuda.grid(2)
    kernelRadius = besarKernel // 2

    if filterX < besarKernel and filterY < besarKernel:
        y = filterY - kernelRadius
        pengali = -1 / (2 * np.pi * delta ** 4)
        kernelY[filterY, filterX] = math.exp(-(y ** 2 / (2 * delta ** 2))) * pengali * y


def gaussian_kernel_derivative_generator(besarKernel, delta):
    
    

    resultX_device = cuda.device_array((besarKernel, besarKernel), dtype=np.float32)
    resultY_device = cuda.device_array((besarKernel, besarKernel), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(besarKernel / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(besarKernel / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    gaussian_kernel_derivative_x_cuda[blockspergrid, threadsperblock](resultX_device, besarKernel, delta)
    gaussian_kernel_derivative_y_cuda[blockspergrid, threadsperblock](resultY_device, besarKernel, delta)

    resultX = resultX_device.copy_to_host()
    resultY = resultY_device.copy_to_host()
    return resultX, resultY


def harris_cpu(src, Gx, Gy, Gxy, thres, k):
    centerKernelGyGx = Gy.shape[1] // 2
    centerKernelGxy = Gxy.shape[1] // 2

    Ix2 = np.zeros((src.shape[0], src.shape[1]))
    Iy2 = np.zeros((src.shape[0], src.shape[1]))
    Ixy = np.zeros((src.shape[0], src.shape[1]))
    IR = np.zeros((src.shape[0], src.shape[1]))

    result = src.copy()

    for i in range(src.shape[1]):
        for j in range(src.shape[0]):
            sX = 0
            sY = 0

            for ik in range(-centerKernelGyGx, centerKernelGyGx + 1):
                ii = i + ik
                for jk in range(-centerKernelGyGx, centerKernelGyGx + 1):
                    jj = j + jk

                    if ii >= 0 and ii < src.shape[1] and jj >= 0 and jj < src.shape[0]:
                        sX += src[jj, ii] * Gx[centerKernelGyGx + jk, centerKernelGyGx + ik]
                        sY += src[jj, ii] * Gy[centerKernelGyGx + jk, centerKernelGyGx + ik]

            Ix2[j, i] = sX * sX
            Iy2[j, i] = sY * sY
            Ixy[j, i] = sX * sY

    for i in range(src.shape[1]):
        for j in range(src.shape[0]):
            sX2 = 0
            sY2 = 0
            sXY = 0

            for ik in range(-centerKernelGxy, centerKernelGxy + 1):
                ii = i + ik
                for jk in range(-centerKernelGxy, centerKernelGxy + 1):
                    jj = j + jk

                    if ii >= 0 and ii < src.shape[1] and jj >= 0 and jj < src.shape[0]:
                        sX2 += Ix2[jj, ii] * Gxy[centerKernelGxy + jk, centerKernelGxy + ik]
                        sY2 += Iy2[jj, ii] * Gxy[centerKernelGxy + jk, centerKernelGxy + ik]
                        sXY += Ixy[jj, ii] * Gxy[centerKernelGxy + jk, centerKernelGxy + ik]

            R = ((sX2 * sY2) - (sXY * sXY)) - k * (sX2 + sY2) ** 2

            if R > thres:
                IR[j, i] = R
            else:
                IR[j, i] = 0

    for y in range(1, IR.shape[0] - 1):
        for x in range(6, IR.shape[1] - 6):

            if (IR[y, x] > IR[y + 1, x] and
                    IR[y, x] > IR[y - 1, x] and
                    IR[y, x] > IR[y, x + 1] and
                    IR[y, x] > IR[y, x - 1] and
                    IR[y, x] > IR[y + 1, x + 1] and
                    IR[y, x] > IR[y + 1, x - 1] and
                    IR[y, x] > IR[y - 1, x + 1] and
                    IR[y, x] > IR[y - 1, x - 1]):
                cv2.circle(result, (x, y), 5, (0, 255, 0), 1)
                result[y, x] = 255

    return result


def non_maximum_suppression(IR, min_distance=1):
    keypoints = []
    distance = 2 * min_distance + 1
    for y in range(min_distance, IR.shape[0] - min_distance):
        for x in range(min_distance, IR.shape[1] - min_distance):
            local_max = np.max(IR[y - min_distance:y + min_distance + 1, x - min_distance:x + min_distance + 1])
            if IR[y, x] == local_max and local_max > 0:
                keypoints.append((x, y))
    return keypoints



@cuda.jit
def harris_kernel(src, Gx, Gy, Gxy, Ix2, Iy2, Ixy, IR, thres, k):
    j, i = cuda.grid(2)  
    height, width = src.shape

    if i < height and j < width:
        
        sX, sY = 0.0, 0.0
        centerKernelGyGx = Gx.shape[1] // 2

        for ik in range(-centerKernelGyGx, centerKernelGyGx + 1):
            ii = i + ik
            if 0 <= ii < height: 
                for jk in range(-centerKernelGyGx, centerKernelGyGx + 1):
                    jj = j + jk
                    if 0 <= jj < width:  
                        sX += src[ii, jj] * Gx[centerKernelGyGx + ik, centerKernelGyGx + jk]
                        sY += src[ii, jj] * Gy[centerKernelGyGx + ik, centerKernelGyGx + jk]

        Ix2[i, j] = sX * sX
        Iy2[i, j] = sY * sY
        Ixy[i, j] = sX * sY

        
        sX2, sY2, sXY = 0.0, 0.0, 0.0
        centerKernelGxy = Gxy.shape[1] // 2

        for ik in range(-centerKernelGxy, centerKernelGxy + 1):
            ii = i + ik
            if 0 <= ii < height:  
                for jk in range(-centerKernelGxy, centerKernelGxy + 1):
                    jj = j + jk
                    if 0 <= jj < width:  
                        sX2 += Ix2[ii, jj] * Gxy[centerKernelGxy + ik, centerKernelGxy + jk]
                        sY2 += Iy2[ii, jj] * Gxy[centerKernelGxy + ik, centerKernelGxy + jk]
                        sXY += Ixy[ii, jj] * Gxy[centerKernelGxy + ik, centerKernelGxy + jk]

        R = ((sX2 * sY2) - (sXY * sXY)) - k * ((sX2 + sY2) * (sX2 + sY2)) 
        
        if R > thres:
            IR[i, j] = R
        else:
            IR[i, j] = 0.0


def draw_keypoints(src, keypoints):
    for x, y in keypoints:
        cv2.circle(src, (x, y), 5, (0, 255, 0), 1)



def harris_cuda(src, Gx, Gy, Gxy, thres, k):
    result = src.copy()
    Ix2_d = cuda.device_array_like(src)
    Iy2_d = cuda.device_array_like(src)
    Ixy_d = cuda.device_array_like(src)
    IR_d = cuda.device_array_like(src)

    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(src.shape[1] / threadsperblock[1]))
    blockspergrid_y = int(np.ceil(src.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    
    harris_kernel[blockspergrid, threadsperblock](src, Gx, Gy, Gxy, Ix2_d, Iy2_d, Ixy_d, IR_d, thres, k)

    
    IR_host = IR_d.copy_to_host()

    keypoints = non_maximum_suppression(IR_host)
    draw_keypoints(result, keypoints)
    return result


class HarrisBenchmark:
    def __init__(self):
        
        self.image_paths = self.generate_images()

    def generate_chessboard(self, size, squares):
        
        image = np.zeros((size, size), dtype=np.uint8)
        square_size = size // squares

        
        for i in range(squares):
            for j in range(squares):
                
                if (i + j) % 2 == 0:
                    top_left = (i * square_size, j * square_size)
                    bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
                    cv2.rectangle(image, top_left, bottom_right, 255, -1)
        return image

    def generate_images(self):
        
        image_sizes = [(128, 128), (256, 256) ,(512, 512)]
        squares = 8  
        image_paths = []

        for width, height in image_sizes:
            
            size = min(width, height)
            size -= size % squares

            
            chessboard_image = self.generate_chessboard(size, squares)
            path = f'chessboard_{size}x{size}.jpg'
            cv2.imwrite(path, chessboard_image)
            image_paths.append(path)

        return image_paths

    def benchmark(self, image_path):
        src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        
        GxyGPU = gaussian_kernel_generator(5, 1.1)
        dGxGPU, dGyGPU = gaussian_kernel_derivative_generator(5, 1.1)

        GxyCPU = gaussian_kernel_generator(7, 3.9)
        dGxCPU, dGyCPU = gaussian_kernel_derivative_generator(7, 1.3)
        
        start_gpu = time.time()
        
        resultGPU = harris_cuda(src, dGxGPU, dGyGPU, GxyGPU, 5000, 0.04)
        end_gpu = time.time()

        
        start_cpu = time.time()
        
        resultCPU = harris_cpu(src, dGxCPU, dGyCPU, GxyCPU, 5000, 0.04)
        end_cpu = time.time()

        gpu_time = end_gpu - start_gpu
        cpu_time = end_cpu - start_cpu

        
        resultGPU = resultGPU
        resultCPU = resultCPU

        
        cv2.imshow('Harris Corners CPU', resultCPU)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Harris Corners GPU', resultGPU)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return gpu_time, cpu_time

    def plot_results(self, gpu_times, cpu_times):
        fig, ax = plt.subplots()
        index = np.arange(len(self.image_paths))
        bar_width = 0.35

        ax.bar(index, cpu_times, bar_width, label='CPU')
        ax.bar(index + bar_width, gpu_times, bar_width, label='GPU')

        ax.set_xlabel('Image Size')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('CPU vs GPU Execution Time')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels([path.split('_')[1].split('x')[0] for path in self.image_paths]) 
        ax.legend()

        plt.show()
    def plot_speedup(self, gpu_times, cpu_times):
        
        speedup = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]

        fig, ax = plt.subplots()
        index = np.arange(len(self.image_paths))
        bar_width = 0.35

        ax.bar(index, speedup, bar_width, label='Speed-Up', color='g')

        ax.set_xlabel('Image Size')
        ax.set_ylabel('Speed-Up Factor')
        ax.set_title('GPU Speed-Up over CPU')
        ax.set_xticks(index)
        ax.set_xticklabels([path.split('_')[1].split('x')[0] for path in self.image_paths])
        ax.legend()

        plt.show()

    def run(self):
        gpu_times = []
        cpu_times = []

        for path in self.image_paths:
            gpu_time, cpu_time = self.benchmark(path)
            gpu_times.append(gpu_time)
            cpu_times.append(cpu_time)
            print(f"Image: {path}, GPU Time: {gpu_time}s, CPU Time: {cpu_time}s")

        self.plot_results(gpu_times, cpu_times)
        self.plot_speedup(gpu_times, cpu_times)



benchmark = HarrisBenchmark()
benchmark.run()