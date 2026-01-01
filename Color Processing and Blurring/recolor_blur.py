import cv2
import numpy as np


def blur_channel(image, mask, kernel_size):
    h, w = image.shape[:2]
    padded_channel = np.zeros((h + kernel_size - 1, w + kernel_size - 1))
    padded_channel[kernel_size // 2: h + kernel_size // 2, kernel_size // 2: w + kernel_size // 2] = image
    blurred_image = np.zeros((h, w))
    for i in range(kernel_size):
        for j in range(kernel_size):
            blurred_image += padded_channel[i:h + i, j:w + j]
    blurred_image //= pow(kernel_size, 2)
    blurred_image[mask] = image[mask]
    return blurred_image


def blur_image(img, mask, kernel_size):
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]
    blurred_blue_channel = blur_channel(blue_channel, mask, kernel_size)
    blurred_green_channel = blur_channel(green_channel, mask, kernel_size)
    blurred_red_channel = blur_channel(red_channel, mask, kernel_size)
    blurred_image = np.zeros((blurred_blue_channel.shape[0], blurred_blue_channel.shape[1], 3))
    blurred_image[:, :, 0] = blurred_blue_channel
    blurred_image[:, :, 1] = blurred_green_channel
    blurred_image[:, :, 2] = blurred_red_channel
    return blurred_image


image = cv2.imread("Flowers.jpg", 1)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]
pink_mask = (hue > 135) & (hue < 180)
hue[pink_mask] += 60
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
result = blur_image(bgr, pink_mask, 19)
cv2.imwrite("res06.jpg", result)
