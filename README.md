# Harris Corner Detector Implementation

This project contains a Python implementation of the Harris Corner Detector, a popular feature detection algorithm used in computer vision. The code is designed to work with both CPU and GPU processing using Numba's CUDA JIT compiler to accelerate computations.

## Features

- Generation of Gaussian and Gaussian derivative kernels on both CPU and GPU.
- Harris Corner Detection implemented for CPU.
- Implementation of non-maximum suppression to refine corner detection.
- Partial implementation of GPU-based Harris Corner Detection (incomplete code).

## Requirements

- Python 3.x
- OpenCV (`cv2`) - for image processing utilities.
- NumPy - for numerical operations on arrays.
- Matplotlib - for plotting (if needed for debugging or visualization).
- Numba - for JIT compilation and CUDA GPU support.

## Setup

To use this code, you must have a Python environment with the required packages installed. You can install the dependencies using `pip`:

```sh
pip install numpy opencv-python matplotlib numba
```

Ensure that you have an NVIDIA GPU with CUDA support to use GPU-accelerated features.

## Usage

The main script `main_harris_code.py` contains function definitions for generating Gaussian kernels, computing the Harris response, and identifying corner points in an image.

To use the Harris Corner Detector, follow these steps:

1. Import the relevant functions from the script.
2. Load an image using OpenCV.
3. Generate Gaussian and Gaussian derivative kernels.
4. Apply the Harris Corner Detection function to the image.
5. Optionally, apply non-maximum suppression to refine the corner points.

Here is an example of how to use the functions in the script:

```python
import cv2
from main_harris_code import gaussian_kernel_generator, harris_cpu, non_maximum_suppression

# Load an image
img = cv2.imread('path_to_your_image.jpg', 0)  # Load in grayscale

# Generate Gaussian kernels
besarKernel = 3
delta = 1.0
Gx, Gy = gaussian_kernel_derivative_generator(besarKernel, delta)

# Apply Harris corner detection
k = 0.04
thres = 1e-2
corners = harris_cpu(img, Gx, Gy, Gxy, thres, k)

# Apply non-maximum suppression
keypoints = non_maximum_suppression(corners, min_distance=1)

# Draw keypoints on the image
for point in keypoints:
    cv2.circle(img, point, 5, (0, 255, 0), 1)

# Display the result
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Notes

- The code provided is not fully tested and may require debugging and completion for the GPU-based functions.
- The performance of the Harris Corner Detection may vary based on the choice of parameters such as kernel size, `delta`, `k`, and `threshold`.

## Contribution

Contributions to improve and complete the code, especially the GPU-based Harris Corner Detection, are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is open-source and available under the MIT License.