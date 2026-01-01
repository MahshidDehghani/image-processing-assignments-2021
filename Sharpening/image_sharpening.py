import math

import cv2
import numpy as np


# Fourier Transform
def dft_2d(image_2d):
    # Compute 2D DFT of a grayscale image
    M, N = image_2d.shape
    F = np.zeros((M, N), complex)
    u = np.arange(0, M, 1)
    v = np.arange(0, N, 1)
    x = u.reshape((M, 1))
    y = v.reshape((N, 1))
    e_yv = np.power(math.e, -2j * math.pi * y * v / N)
    e_xu = np.power(math.e, -2j * math.pi * x * u / M)
    for i in range(M):
        F[i] = np.dot(e_yv, image_2d[i])
    for j in range(N):
        F[:, j] = np.dot(e_xu, F[:, j])
    return F / (M * N)


def dft(image):
    # Compute DFT of grayscale or color image
    shape = image.shape
    if len(shape) == 2:
        return dft_2d(image)
    dft = np.zeros(shape, complex)
    for color in range(shape[2]):
        dft[:, :, color] = dft_2d(image[:, :, color])
    return dft


def idft_2d(F):
    # Compute 2D inverse DFT for grayscale image
    M, N = F.shape
    image = np.zeros((M, N), complex)
    u = np.arange(0, M, 1)
    v = np.arange(0, N, 1)
    x = u.reshape((M, 1))
    y = v.reshape((N, 1))
    e_yv = np.power(math.e, 2j * math.pi * y * v / N)
    e_xu = np.power(math.e, 2j * math.pi * x * u / M)
    for j in range(N):
        image[:, j] = np.dot(e_xu, F[:, j])
    for i in range(M):
        image[i] = np.dot(e_yv, image[i])
    return image


def idft(F):
    # Compute inverse DFT for grayscale or color image
    shape = F.shape
    if len(shape) == 2:
        return idft_2d(F)
    image = np.zeros(shape, complex)
    for color in range(shape[2]):
        image[:, :, color] = idft_2d(F[:, :, color])
    return image


def shift_2d(dft):
    # Shift zero-frequency component to the center for grayscale image
    h, w, = dft.shape
    a_h, a_w = (h/2 - h//2), (w/2 - w//2)
    q1, q2 = dft[0:int(h/2 + a_h), 0:int(w/2 + a_w)], dft[int(h/2 + a_h):h, 0:int(w/2 + a_w)]
    q3, q4 = dft[0:int(h/2 + a_h), int(w/2 + a_w):w], dft[int(h/2 + a_h):h, int(w/2 + a_w):w]
    shift_F = np.zeros((h, w), complex)
    shift_F[int(h/2 - a_h):h, int(w/2 - a_w):w], shift_F[0:int(h/2 - a_h), 0:int(w/2 - a_w)] = q1, q4
    shift_F[int(h/2 - a_h):h, 0:int(w/2 - a_w)], shift_F[0:int(h/2 - a_h), int(w/2 - a_w):w] = q3, q2
    return shift_F


def shift(dft):
    # Shift zero-frequency to center for grayscale or color image
    shape = dft.shape
    if len(shape) == 2:
        return shift_2d(dft)
    shift_F = np.zeros(shape, complex)
    for i in range(shape[2]):
        shift_F[:, :, i] = shift_2d(dft[:, :, i])
    return shift_F


def ishift_2d(shift_F):
    # Inverse shift for grayscale image
    h, w, = shift_F.shape
    a_h, a_w = (h / 2 - h // 2), (w / 2 - w // 2)
    q1, q4 = shift_F[int(h / 2 - a_h):h, int(w / 2 - a_w):w], shift_F[0:int(h / 2 - a_h), 0:int(w / 2 - a_w)]
    q3, q2 = shift_F[int(h / 2 - a_h):h, 0:int(w / 2 - a_w)], shift_F[0:int(h / 2 - a_h), int(w / 2 - a_w):w]
    F = np.zeros((h, w), complex)
    F[0:int(h / 2 + a_h), 0:int(w / 2 + a_w)], F[int(h / 2 + a_h):h, 0:int(w / 2 + a_w)] = q1, q2
    F[0:int(h / 2 + a_h), int(w / 2 + a_w):w], F[int(h / 2 + a_h):h, int(w / 2 + a_w):w] = q3, q4
    return F


def ishift(shift_F):
    # Inverse shift for grayscale or color image
    shape = shift_F.shape
    if len(shape) == 2:
        return shift_2d(shift_F)
    F = np.zeros(shape, complex)
    for i in range(shape[2]):
        F[:, :, i] = ishift_2d(shift_F[:, :, i])
    return F


# Frequency-domain Filters
def create_low_pass_gaussian_filter(dim, D0):
    M, N = dim
    m_diff = M / 2 - M // 2
    n_diff = N / 2 - N // 2
    u = np.arange(start=m_diff, stop=M + m_diff, step=1)
    v = np.arange(start=n_diff, stop=N + n_diff, step=1)
    if M%2 == 0:
        u[u >= M / 2] -= (M - 1)
    else:
        u[u > M / 2] -= M
    if N%2 == 0:
        v[v >= N / 2] -= (N - 1)
    else:
        v[v > N / 2] -= N
    [V, U] = np.meshgrid(v, u)
    D = U * U + V * V
    D /= (2 * (D0 ** 2))
    H = np.power(math.e, D)
    H = cv2.normalize(src=H, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return H


def create_high_pass_gaussian_filter(dim, D0):
    return 1 - create_low_pass_gaussian_filter(dim, D0)


def apply_frequency_filter(shift_dft, filter):
    # Apply filter in frequency domain (element-wise multiplication)
    shape = shift_dft.shape
    if len(shape) == 2:
        return shift_dft * filter
    filtered = np.zeros(shape, complex)
    for i in range(shape[2]):
        filtered[:, :, i] = shift_dft[:, :, i] * filter
    return filtered


# Spatial Filters
def g(sigma, x, y) :
    return (1/(2 * math.pi * pow(sigma, 2))) * math.exp(-(pow(x, 2) + pow(y, 2))/(2 * pow(sigma, 2)))


def LoG(sigma, x, y):
    a = (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))
    return (1 / (math.pi * pow(sigma, 4))) * (a - 1) * math.exp(-a)


def create_gaussian_kernel(sigma, kernel_size):
    g_filter = np.zeros((kernel_size, kernel_size))
    a = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            g_filter[i, j] = g(sigma, i - a, j - a)
    g_filter = g_filter / np.sum(g_filter)
    return g_filter


def create_log_kernel(sigma, kernel_size):
    l_filter = np.zeros((kernel_size, kernel_size))
    a = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            l_filter[i, j] = LoG(sigma, i - a, j - a)
    return l_filter


def smooth_2d(image_2d, g_filter):
    # Convolution for 2D grayscale image
    kernel_size = g_filter.shape[0]
    a = kernel_size // 2
    height = image_2d.shape[0]
    width = image_2d.shape[1]
    large_channel = np.zeros((height + kernel_size - 1, width + kernel_size - 1))
    large_channel[a: height + a, a: width + a] = image_2d
    smoothed_image = np.zeros((height, width))
    for i in range(kernel_size):
        for j in range(kernel_size):
            smoothed_image += large_channel[i:height + i, j:width + j] * g_filter[i, j]
    smoothed_image[0:height, 0:a] = image_2d[0:height, 0:a]
    smoothed_image[0:height, width - a: width] = image_2d[0:height, width - a: width]
    smoothed_image[0:a, 0:width] = image_2d[0:a, 0:width]
    smoothed_image[height-a:height, 0:width] = image_2d[height-a:height, 0:width]
    return smoothed_image


def smooth(image, g_filter):
    # Convolution for grayscale or color image
    shape = image.shape
    if len(shape) == 2:
        return smooth_2d(image, g_filter)
    smoothed_image = np.zeros(shape)
    for color in range(shape[2]):
        smoothed_image[:, :, color] = smooth_2d(image[:, :, color], g_filter)
    return smoothed_image


def apply_laplacian_of_gaussian_filter_2d(image_2d, log_filter):
    kernel_size = log_filter.shape[0]
    a = kernel_size // 2
    height = image_2d.shape[0]
    width = image_2d.shape[1]
    filtered_image = np.zeros((height, width))
    for i in range(kernel_size):
        for j in range(kernel_size):
            h = height - kernel_size + 1 + i
            w = width - kernel_size + 1 + j
            filtered_image[a: height - a, a: width - a] += image_2d[i:h, j:w] * log_filter[i, j]
    return filtered_image


def apply_laplacian_of_gaussian_filter(image, l_filter):
    shape = image.shape
    if len(shape) == 2:
        return apply_laplacian_of_gaussian_filter_2d(image, l_filter)
    laplacian_image = np.zeros(shape)
    for color in range(shape[2]):
        laplacian_image[:, :, color] = apply_laplacian_of_gaussian_filter_2d(image[:, :, color], l_filter)
    return laplacian_image


def compute_laplacian_dft(image):
    M, N = image.shape[:2]
    m_diff = M / 2 - M // 2
    n_diff = N / 2 - N // 2
    u = np.arange(start=m_diff, stop=M + m_diff, step=1)
    v = np.arange(start=n_diff, stop=N + n_diff, step=1)
    if M % 2 == 0:
        u[u >= M / 2] -= (M - 1)
    else:
        u[u > M / 2] -= M
    if N % 2 == 0:
        v[v >= N / 2] -= (N - 1)
    else:
        v[v > N / 2] -= N
    [V, U] = np.meshgrid(v, u)
    D = U * U + V * V
    D = 4 * np.pi * D
    shift_dft = dft(image)
    return apply_frequency_filter(shift_dft, D)


def visualize_kernel(kernel, kernel_size, pixel_size):
    new = np.zeros((kernel_size * pixel_size, kernel_size * pixel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            new[pixel_size * i:pixel_size * i + pixel_size, pixel_size * j:pixel_size * j + pixel_size] = kernel[i, j]
    return new


image = cv2.imread("flowers.blur.png")
image_full_res = cv2.imread("flowers.blur.full.res.png")

# first method
sigma = 1
kernel_size = 5
alfa = 5
g_filter = create_gaussian_kernel(sigma, kernel_size)
gaussian_filter_image = visualize_kernel(cv2.normalize(src=g_filter, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), kernel_size, 100)
smoothed_image = smooth(image, g_filter)
unsharp_mask = image - smoothed_image
sharped_image = image + alfa * unsharp_mask
cv2.imwrite("res01.jpg", gaussian_filter_image)
cv2.imwrite("res02.jpg", smoothed_image)
cv2.imwrite("res03.jpg", cv2.normalize(src=unsharp_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
cv2.imwrite("res04.jpg", sharped_image)


# second method
sigma = 1
kernel_size = 9
k = 4
LoG_filter = create_log_kernel(sigma, kernel_size)
laplacian_of_gaussian_filter_image = visualize_kernel(cv2.normalize(src=LoG_filter, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), kernel_size, 100)
unsharp_mask = apply_laplacian_of_gaussian_filter(image, LoG_filter)
sharped_image = image - k * unsharp_mask
cv2.imwrite("res05.jpg", laplacian_of_gaussian_filter_image)
cv2.imwrite("res06.jpg", cv2.normalize(src=unsharp_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
cv2.imwrite("res07.jpg", sharped_image)


# third method
D0 = 700
dim = (image.shape[0], image.shape[1])
k = 5
dft_img = shift(dft(image))
high_pass_filter = create_high_pass_gaussian_filter(dim, D0)
highpassed_dft = apply_frequency_filter(dft_img, (1 + k * high_pass_filter))
sharped_image = np.real(idft(ishift(highpassed_dft)))
cv2.imwrite("res08.jpg", cv2.normalize(np.log(np.abs(dft_img)), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
cv2.imwrite("res09.jpg", (high_pass_filter * 255).astype('uint8'))
cv2.imwrite("res10.jpg", cv2.normalize(np.log(np.abs(highpassed_dft)), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
cv2.imwrite("res11.jpg", sharped_image)


# fourth method
k = 0.00001
dft_laplacian = compute_laplacian_dft(image)
laplacian = np.real(idft((dft_laplacian)))
sharped_image = image + k * laplacian
cv2.imwrite("res12.jpg", cv2.normalize(np.real(dft_laplacian), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
cv2.imwrite("res13.jpg", laplacian)
cv2.imwrite("res14.jpg", sharped_image)
