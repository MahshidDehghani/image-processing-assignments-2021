import time
import cv2
import numpy as np


def smooth_with_opencv(channel):
    return cv2.filter2D(channel, -1, BOX_FILTER, cv2.BORDER_CONSTANT)


def smooth_with_loops(channel, h, w):
    smoothed_channel = np.zeros((h, w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            smoothed_channel[i, j] = np.around(np.sum(channel[i - 1:i + 2, j - 1:j + 2] * BOX_FILTER))
    return smoothed_channel


def smooth_vectorized(channel, h, w):
    smoothed_channel = np.zeros((h, w))
    smoothed_channel[1:h - 1, 1:w - 1] = np.around(
        channel[0:h - 2, 0:w - 2] * BOX_FILTER[0, 0] + \
        channel[0:h - 2, 1:w - 1] * BOX_FILTER[0, 1] + \
        channel[0:h - 2, 2:w] * BOX_FILTER[0, 2] + \
        channel[1:h - 1, 0:w - 2] * BOX_FILTER[1, 0] + \
        channel[1:h - 1, 1:w - 1] * BOX_FILTER[1, 1] + \
        channel[1:h - 1, 2:w] * BOX_FILTER[1, 2] + \
        channel[2:h, 0:w - 2] * BOX_FILTER[2, 0] + \
        channel[2:h, 1:w - 1] * BOX_FILTER[2, 1] + \
        channel[2:h, 2:w] * BOX_FILTER[2, 2])
    return smoothed_channel


def apply_smoothing(image, method_name):
    h, w = image.shape[:2]
    if method_name == "opencv":
        return smooth_with_opencv(image)
    smoothed_image = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # Loop over color channels
        channel = image[:, :, c]
        if method_name == "loops":
            smoothed_image[:, :, c] = smooth_with_loops(channel, h, w)
        else:
            smoothed_image[:, :, c] = smooth_vectorized(channel, h, w)
    return smoothed_image.astype(np.uint8)


BOX_FILTER = np.ones((3, 3), np.float32) / 9
image = cv2.imread("Pink.jpg", 1)
h, w = image.shape[:2]
start_time = time.time()
res7 = apply_smoothing(image, method_name="opencv")

cv2.imwrite("res07.jpg", res7)
print("method1 : %s seconds" % (time.time() - start_time))


start_time = time.time()
res8 = apply_smoothing(image, method_name="loops")
cv2.imwrite("res08.jpg", res8)
print("method2 : %s seconds" % (time.time() - start_time))


start_time = time.time()
res9 = apply_smoothing(image, method_name="vectorized")
cv2.imwrite("res09.jpg", res9)
print("method3 : %s seconds" % (time.time() - start_time))

all_methods_equal = np.array_equal(res7[1:h - 1, 1:w - 1, :], res8[1:h - 1, 1:w - 1, :]) & \
           np.array_equal(res7[1:h - 1, 1:w - 1, :], res9[1:h - 1, 1:w - 1, :])
print("Three methods have same results ->", all_methods_equal)