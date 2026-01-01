import cv2
import numpy as np


def transpose(img):
    h, w, c = img.shape
    transposed_image = np.zeros((w, h, c), dtype='int')
    for i in range(img.shape[0]):
        transposed_image[:, i] = img[i]
    return transposed_image


def warp(img, projective_matrix, h, w):
    transposed_image = transpose(img)
    warped_image = np.zeros((h, w, transposed_image.shape[2]))
    pixel_filled_mask = np.zeros((h, w)).astype('int')
    for i in range(transposed_image.shape[0]):
        for j in range(transposed_image.shape[1]):
            projected_point = np.dot(projective_matrix, [i, j, 1])
            x, y, _ = (projected_point / projected_point[2] + 0.5).astype(int)
            if 0 <= x < h:
                if 0 <= y < w:
                    warped_image[x, y] = transposed_image[i, j]
                    pixel_filled_mask[x, y] = 1
    unfilled_pixel_indices = np.array(np.where(pixel_filled_mask == 0))
    for i in range(unfilled_pixel_indices[0].size):
        x, y = unfilled_pixel_indices[0, i], unfilled_pixel_indices[1, i]
        neighbor_count = 0
        if x - 1 >= 0:
            warped_image[x, y] += warped_image[x - 1, y]
            neighbor_count += 1
        if y - 1 >= 0:
            warped_image[x, y] += warped_image[x , y - 1]
            neighbor_count += 1
        if x + 1 < h :
            warped_image[x, y] += warped_image[x + 1, y]
            neighbor_count += 1
        if y + 1 < w:
            warped_image[x, y] += warped_image[x, y + 1]
            neighbor_count += 1
        if neighbor_count > 0:
            warped_image[x, y] //= neighbor_count
    return transpose(warped_image)


books = cv2.imread("books.jpg")

# Discrete Book
h, w = 211, 286
src_points = np.float32([[365, 745], [157, 711], [412, 463], [204, 429]])
books[232, 473, :] = 0
dst_points = np.float32([[0, 0], [h - 1, 0], [0, w - 1], [h - 1, w - 1]])
homography = cv2.getPerspectiveTransform(src_points, dst_points)
# img_output = cv2.warpPerspective(books, homography, (h, w))
# cv2.imwrite("Discrete_Book.jpg", img_output)
res = warp(books, homography, h, w)
cv2.imwrite("res16.jpg", res)

# Graph Book
h, w = 197, 306
src_points = np.float32([[670, 208], [599, 394], [383, 103], [318, 289]])
books[232, 473, :] = 0
dst_points = np.float32([[0, 0], [h - 1, 0], [0, w - 1], [h - 1, w - 1]])
homography = cv2.getPerspectiveTransform(src_points, dst_points)
# img_output = cv2.warpPerspective(books, homography, (h, w))
# cv2.imwrite("Graph_Book.jpg", img_output)
res = warp(books, homography, h, w)
cv2.imwrite("res17.jpg", res)

# Image Science Book
h, w = 245, 372
src_points = np.float32([[817, 973], [610, 1103], [621, 657], [413, 787]])
books[232, 473, :] = 0
dst_points = np.float32([[0, 0], [h - 1, 0], [0, w - 1], [h - 1, w - 1]])
homography = cv2.getPerspectiveTransform(src_points, dst_points)
# img_output = cv2.warpPerspective(books, homography, (h, w))
# cv2.imwrite("Image_Science_Book.jpg", img_output)
res = warp(books, homography, h, w)
cv2.imwrite("res18.jpg", res)
