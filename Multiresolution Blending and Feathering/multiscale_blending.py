import cv2
import numpy as np


def gaussian_stack(num_levels, image, kernel_size, base_sigma):
    g_stack = []
    prev = image
    for l in range(num_levels):
        level_sigma = np.power(base_sigma, l)
        if l > 0:
            prev = g_stack[l - 1]
        g_stack.append(cv2.GaussianBlur(prev, ksize=(kernel_size, kernel_size), sigmaX=level_sigma, sigmaY=level_sigma))
    # del g_stack[0]
    return g_stack


def laplacian_stack(num_levels, image, g_stack):
    l_stack = [image.astype('int') - g_stack[0].astype('int')]
    for l in range(0, num_levels - 2):
        l_stack.append(g_stack[l].astype('int') - g_stack[l + 1].astype('int'))
    l_stack.append(g_stack[num_levels - 1])
    return l_stack


def stack_blend(num_levels, im1, im2, horizontal_split=True, kernel_size=45, base_sigma=1.2):
    h, w = im1.shape[:2]
    mask = np.zeros((h, w, 3))
    if horizontal_split:
        mask[:h // 2, :, :] = 1
    else:
        mask[:, :w//2, :] = 1
    mask_stack = gaussian_stack(num_levels, mask, kernel_size=kernel_size, base_sigma=2 * base_sigma)

    im1_g_stack = gaussian_stack(num_levels, im1, kernel_size, base_sigma)
    im1_l_stack = laplacian_stack(num_levels, im1, im1_g_stack)

    im2_g_stack = gaussian_stack(num_levels, im2, kernel_size, base_sigma)
    im2_l_stack = laplacian_stack(num_levels, im2, im2_g_stack)

    res = np.zeros((h, w, 3)).astype('uint8')
    blended_stack = []
    for l in range(num_levels):
        im1_l = im1_l_stack[l]
        im2_l = im2_l_stack[l]
        mask1 = mask_stack[l]
        mask2 = np.ones((h, w, 3)) - mask1
        blended = (np.multiply(im1_l, mask1) + np.multiply(im2_l, mask2)).astype('uint8')
        blended_stack.append(blended)
        res += blended
        # cv2.imwrite("im1_" + str(l) + ".jpg", cv2.normalize(im1_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8'))
        # cv2.imwrite("im2_" + str(l) + ".jpg", cv2.normalize(im2_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8'))
        # cv2.imwrite("m1_" + str(l) + ".jpg", cv2.normalize(mask1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8'))
        # cv2.imwrite("m2_" + str(l) + ".jpg", cv2.normalize(mask2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8'))
        # cv2.imwrite("blended_" + str(l) + ".jpg", cv2.normalize(blended, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8'))
    return res


def laplacian_gaussian_pyramid(num_levels, image, kernel_size, sigma):
    g_pyramid = []
    l_pyramid = []
    current_level_image = image
    for l in range(num_levels):
        I = cv2.GaussianBlur(current_level_image.astype('uint8'), ksize=(kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
        h = current_level_image.astype('uint8') - I.astype('int')
        g_pyramid.append(I)
        l_pyramid.append(h)
        current_level_image = cv2.resize(I, (0, 0), fx=0.5, fy=0.5)
    l_pyramid.append(current_level_image)
    return g_pyramid, l_pyramid


def create_feathered_blend_masks(h, w, transition_fraction, center_offset, horizontal_split):
    mask1 = np.zeros((h, w, 3))
    mask2 = np.zeros((h, w, 3))
    if horizontal_split:
        transition_center = h // 2 + center_offset
        transition_size = int(transition_fraction * h)
        half_transition_width = transition_size // 2
        mask1[:transition_center - half_transition_width + 1, :, :] = 1
        mask2[transition_center + half_transition_width:, :, :] = 1
        for i in range(transition_size):
            mask1[transition_center - half_transition_width + i, :, :] = (transition_size - i - 1) / transition_size
            mask2[transition_center + half_transition_width - i, :, :] = (transition_size - i - 1) / transition_size
    else:
        transition_center = w // 2 + center_offset
        transition_size = int(transition_fraction * w)
        half_transition_width = transition_size // 2
        mask1[:, :transition_center - half_transition_width + 1, :] = 1
        mask2[:, transition_center + half_transition_width:, :] = 1
        for i in range(transition_size):
            mask1[:, transition_center - half_transition_width + i, :] = (transition_size - i - 1) / transition_size
            mask2[:, transition_center + half_transition_width - i, :] = (transition_size - i - 1) / transition_size
    return mask1, mask2


def laplacian_pyramid_blend(num_levels, im1, im2, transition_fraction, center_offset=0, horizontal_split=True, kernel_size=5, sigma=0):
    im1_g_pyramid, im1_l_pyramid = laplacian_gaussian_pyramid(num_levels, im1, kernel_size, sigma)
    im2_g_pyramid, im2_l_pyramid = laplacian_gaussian_pyramid(num_levels, im2, kernel_size, sigma)
    blended = np.zeros_like(im1_l_pyramid[num_levels])
    blended_pyramid = []
    for l in range(num_levels, -1, -1):
        im1_l = im1_l_pyramid[l]
        im2_l = im2_l_pyramid[l]
        h, w = im1_l.shape[:2]
        mask1, mask2 = create_feathered_blend_masks(h, w, transition_fraction / (np.power(2, num_levels - l)), center_offset, horizontal_split)
        blended += (np.multiply(im1_l, mask1) + np.multiply(im2_l, mask2)).astype('uint8')
        if l != 0:
            next_level_h, next_level_w = im1_l_pyramid[l - 1].shape[:2]
            blended = cv2.resize(blended, (next_level_w, next_level_h))
            blended.astype('int')
        blended_pyramid.append(blended)
    return blended_pyramid[-1].astype('uint8')


image1 = cv2.imread("res08.jpg")
image2 = cv2.imread("res09.jpg")
h, w = image2.shape[:2]
image1 = cv2.resize(image1, (w, h))
mask = np.zeros_like(image2)
mask[:, :w//2] = 1
res = laplacian_pyramid_blend(num_levels=5, im1=image1, im2=image2, transition_fraction=0.9)
cv2.imwrite("pyramid_blend.jpg", res)
res = stack_blend(num_levels=12, im1=image1, im2=image2)
cv2.imwrite("res10.jpg", res)

