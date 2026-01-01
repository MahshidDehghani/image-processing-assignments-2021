import cv2
import numpy as np
import time


def histogram(image, L):
    shape = image.shape
    hist = np.zeros((L))
    for i in range(shape[0]):
        for j in range(shape[1]):
            intensity = int(image[i][j])
            hist[intensity] += 1
    return hist


def normalize_histogram(image, L):
    size = image.size
    hist = histogram(image, L)
    hist[:] = hist[:] / size
    return hist


def cdf(image, L):
    size = image.size
    hist = histogram(image, L)
    for i in range(1, L):
        hist[i] += hist[i-1]
    hist[:] = hist[:] / size
    return hist


def equalize_histogram(image, L):
    img = np.copy(image)
    S = cdf(img, L)
    S[:] = S[:] * (L - 1)
    S = np.around(S)
    img = S[img]
    return img


def stretch_histogram(image, L):
    img = np.copy(image)
    min = np.min(img)
    max = np.max(img)
    diff = max - min
    img = np.around((img - min) / diff * (L - 1))
    return img


def log_transformation(image, L):
    x = np.any(image < 0)
    return np.round(((L-1)/np.log(L)) * np.log(1 + image))


def enhance(image_address, L, method_name, write=True):
    image = cv2.imread(image_address, 1)
    enhanced_image = np.zeros(image.shape)
    blue_channel = image[:,:,0]
    green_channel = image[:,:,1]
    red_channel = image[:,:,2]
    if method_name == "equalize_histogram" or method_name == "equalize" or method_name == "e":
        enhanced_image[:, :, 0] = equalize_histogram(blue_channel, L)
        enhanced_image[:, :, 1] = equalize_histogram(green_channel, L)
        enhanced_image[:, :, 2] = equalize_histogram(red_channel, L)
    elif method_name == "stretch_histogram" or method_name == "stretch" or method_name == "s":
        enhanced_image[:, :, 0] = stretch_histogram(blue_channel, L)
        enhanced_image[:, :, 1] = stretch_histogram(green_channel, L)
        enhanced_image[:, :, 2] = stretch_histogram(red_channel, L)
    else:
        enhanced_image[:, :, 0] = log_transformation(blue_channel, L)
        enhanced_image[:, :, 1] = log_transformation(green_channel, L)
        enhanced_image[:, :, 2] = log_transformation(red_channel, L)
    enhanced_image_name = method_name[0] + "_" + image_address
    if write:
        cv2.imwrite(enhanced_image_name, enhanced_image)
    return enhanced_image, enhanced_image_name


image_name = "Enhance1.JPG"
L = 256
equalized_image, _ = enhance(image_name, L, method_name="equalize_histogram", write=False)
cv2.imwrite("equalized_" + image_name, equalized_image)

log_transformed_image, log_transformed_image_name = enhance(image_name, L, method_name="log_transformation")

stretched_image, _ = enhance(log_transformed_image_name, L, method_name="stretch_histogram", write=False)
cv2.imwrite("stretched_log_" + image_name, stretched_image)

result = (equalized_image + stretched_image)//2
cv2.imwrite("res01.jpg", result)