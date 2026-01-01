import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def compute_histogram(image, L):
    shape = image.shape
    histogram = np.zeros((L))
    for i in range(shape[0]):
        for j in range(shape[1]):
            intensity = int(image[i][j])
            histogram[intensity] += 1
    return histogram


def compute_normalized_histogram(image, L):
    size = image.size
    histogram = compute_histogram(image, L)
    histogram[:] = histogram[:] / size
    return histogram


def cdf(image, L):
    size = image.size
    histogram = compute_histogram(image, L)
    for i in range(1, L):
        histogram[i] += histogram[i - 1]
    histogram[:] = histogram[:] / size
    return histogram


def histogram_match_channel(image, target_histogram, L):
    img = np.copy(image)
    S = cdf(img, L)
    S[:] = S[:] * (L - 1)
    S = np.around(S).astype(int)

    G = np.copy(target_histogram)
    for i in range(1, L):
        G[i] += G[i - 1]
    G[:] = G[:] * (L - 1)
    G = np.around(G).astype(int)

    inverse_G = {}
    for i in range(L):
        if G[i] not in inverse_G.keys():
            inverse_G[G[i]] = i

    for i in range(L):
        if S[i] in inverse_G.keys():
            S[i] = inverse_G[int(S[i])]
        else:
            lower_match = G[G < int(S[i])].max()
            upper_match = G[G > int(S[i])].min()
            if upper_match - S[i] >= S[i] - lower_match:
                S[i] = lower_match
                inverse_G[S[i]] = lower_match
            else:
                S[i] = upper_match
                inverse_G[S[i]] = upper_match
    img = S[img]
    return img


def histogram_match_image(source_image, target_image, L):
    matched_image = np.zeros(source_image.shape)
    blue_channel = source_image[:, :, 0]
    green_channel = source_image[:, :, 1]
    red_channel = source_image[:, :, 2]
    blue_hist = compute_normalized_histogram(target_image[:, :, 0], L)
    green_hist = compute_normalized_histogram(target_image[:, :, 1], L)
    red_hist = compute_normalized_histogram(target_image[:, :, 2], L)
    matched_image[:, :, 0] = histogram_match_channel(blue_channel, blue_hist, L)
    matched_image[:, :, 1] = histogram_match_channel(green_channel, green_hist, L)
    matched_image[:, :, 2] = histogram_match_channel(red_channel, red_hist, L)
    return matched_image


L = 256
dark_img = cv2.imread("Dark.jpg", 1)
pink_img = cv2.imread("Pink.jpg", 1)
result = histogram_match_image(dark_img, pink_img, L)
cv2.imwrite("res11.jpg", result)

blue_histogram = compute_normalized_histogram(result[:, :, 0], L)
green_histogram = compute_normalized_histogram(result[:, :, 1], L)
red_histogram = compute_normalized_histogram(result[:, :, 2], L)

plt.figure(figsize=(15.2, 4.8))
plt.subplot(131)
plt.plot(blue_histogram, 'b')
plt.title("Blue")
plt.subplot(132)
plt.plot(green_histogram, 'g')
plt.title("Green")
plt.subplot(133)
plt.plot(red_histogram, 'r')
plt.title("Red")
plt.savefig("res10.jpg")
