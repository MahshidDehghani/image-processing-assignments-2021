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
    # Compute inverse DFT for grayscale image
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


def ideal_low_pass_filter(dim, D0):
    M, N = dim
    m_diff = M / 2 - M // 2
    n_diff = N / 2 - N // 2
    u = np.arange(start=m_diff, stop=M + m_diff, step=1) - M/2
    v = np.arange(start=n_diff, stop=N + n_diff, step=1) - N/2
    if M%2 == 0:
        u[0:int(M/2)] += 1
    if N%2 == 0:
        v[0:int(N/2)] += 1
    u = np.abs(u)
    v = np.abs(v)
    ideal = np.zeros(dim)
    [V, U] = np.meshgrid(v, u)
    D = U * U + V * V
    ideal[D <= D0 ** 2] = 1
    return ideal


def ideal_high_pass_filter(dim, D0):
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


def average(img1, img2, weight):
    shape1 = img1.shape
    shape2 = img2.shape
    if shape1 == shape2:
        return weight * img1 + (1 - weight) * img2
    else:
        if len(shape2) == 2:
            avg = np.zeros(shape1, complex)
            for i in range(shape1[2]):
                avg[:, :, i] = weight * img1[:, :, i] + (1 - weight) * img2
            return avg
        avg = np.zeros(shape2, complex)
        for i in range(shape2[2]):
            avg[:, :, i] = weight * img1 + (1 - weight) * img2[:, :, i]
        return avg


def hybrid(dim, near_img, far_img, s, r, weight, far_scale):
    dft_near = dft(near_img)
    dft_far = dft(far_img)
    cv2.imwrite("res23_dft_near.jpg", cv2.normalize(np.log(np.abs(dft_near)), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
    cv2.imwrite("res24_dft_far.jpg", cv2.normalize(np.log(np.abs(dft_far)), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))

    shift_dft_near = shift(dft_near)
    shift_dft_far = shift(dft_far)

    high_pass_filter = create_high_pass_gaussian_filter(dim, r)
    low_pass_filter = create_low_pass_gaussian_filter(dim, s)
    save_img = (high_pass_filter * 255).astype('uint8')
    cv2.imwrite("res25_highpass_" + str(r) + ".jpg", save_img)
    save_img = (low_pass_filter * 255).astype('uint8')
    cv2.imwrite("res26_lowpass_" + str(s) + ".jpg", save_img)

    fil_shift_dft_near = apply_frequency_filter(shift_dft_near, high_pass_filter)
    fil_shift_dft_far = apply_frequency_filter(shift_dft_far, low_pass_filter)

    save_img = cv2.normalize(np.abs(fil_shift_dft_near), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite("res27_highpassed.jpg", save_img)
    save_img = cv2.normalize(np.abs(fil_shift_dft_far), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite("res28_lowpassed.jpg", save_img)

    weighted_average = average(fil_shift_dft_near, fil_shift_dft_far, weight)
    save_img = cv2.normalize(np.log(np.abs(weighted_average)), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite("res29_hybrid.jpg", save_img)

    hybrid = np.real(idft(ishift(weighted_average)))
    cv2.imwrite("res30_hybrid_near.jpg", hybrid)
    save_img = cv2.resize(hybrid, (0, 0), fx=far_scale, fy=far_scale)
    cv2.imwrite("res31_hybrid_far.jpg", save_img)
    return hybrid


dog = cv2.imread("res19_near.jpg")
cat = cv2.imread("res20_far.jpg", 0)
cat_h, cat_w = cat.shape[:2]
dim = (cat_h, cat_w)
dog = cv2.resize(dog, (cat_w, cat_h),  interpolation = cv2.INTER_AREA)
cv2.imwrite("res21_near.jpg", dog)
cv2.imwrite("res22_far.jpg", cat)
r = 50
s = 50
weight = 1 / 2
far_scale = 0.2
h = hybrid(dim, dog, cat, s, r, weight, far_scale)
cv2.imwrite("res30_hybrid_near.jpg", cv2.resize(h, (0, 0), fx=2, fy=2))





