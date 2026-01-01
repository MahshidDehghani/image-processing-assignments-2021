import numpy as np
import cv2
import time


def compute_ncc_2d(image_2d, template_2d):
    img_h, img_w = image_2d.shape
    temp_h, temp_w = template_2d.shape
    ccofe_h, ccofe_w = img_h - temp_h + 1, img_w - temp_w + 1
    t_mean = np.sum(template_2d) / (temp_h * temp_w)
    i_mean = np.zeros((ccofe_h, ccofe_w))
    for i in range(temp_h):
        for j in range(temp_w):
            i_mean += image_2d[i:ccofe_h + i, j:ccofe_w + j]
    i_mean = i_mean / (temp_h * temp_w)
    ccoeff = np.zeros((ccofe_h, ccofe_w))
    for i in range(temp_h):
        for j in range(temp_w):
            ccoeff += (template_2d[i, j] - t_mean) * (image_2d[i:ccofe_h + i, j:ccofe_w + j] - i_mean)
    return ccoeff


def compute_ncc(image, template):
    shape = image.shape
    if len(shape) == 2:
        return compute_ncc_2d(image, template)
    ccoeff = compute_ncc_2d(image[:, :, 0], template[:, :, 0])
    for color in range(1, shape[2]):
        ccoeff += compute_ncc_2d(image[:, :, color], template[:, :, color])
    ccoeff /= shape[2]
    return ccoeff


def extract_template_patch(patch):
    h, w = patch.shape[:2]
    t_xs, t_h = h // 4, h // 16
    t_ys, t_w = w // 2 + w // 60, w // 30
    cropped_template = patch[t_xs:t_xs + t_h, t_ys:t_ys + t_w, :]
    return cropped_template


image = cv2.imread("Greek-ship.jpg")
resized_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
template = cv2.imread("patch.png")
cropped_template = extract_template_patch(template)
cv2.imwrite("temp.png", cropped_template)


ccoeff = compute_ncc(resized_image, cropped_template)
ccoeff = cv2.normalize(src=ccoeff, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imwrite("ccoeff.jpg", ccoeff)

mask = np.zeros(ccoeff.shape)
mask[ccoeff > 180] = 255
white_pixel_counts = np.mean(mask, axis=0)
WHITE_THRESHOLD = 5
white_pixel_counts[white_pixel_counts >= WHITE_THRESHOLD] = 255
white_pixel_counts[white_pixel_counts < WHITE_THRESHOLD] = 0
columns_with_matches = np.asarray(np.where(white_pixel_counts > 0))
list = []
for y in columns_with_matches[0, :]:
    x = np.asarray(np.where(mask[:,y] > 0))[0,:]
    if len(list) != 0 and white_pixel_counts[y - 1] > 0:
        box_coords = list[-1]
        if box_coords[1] > x[0]:
            box_coords[1] = x[0]
        if box_coords[2] < x[-1]:
            box_coords[2] = x[-1]
    else:
        box_coords = np.array([y, x[0], x[-1]])
        list.append(box_coords)


# draw red box
box_color = (0, 0, 255)
box_thickness = 3
for i in range(len(list)):
    box_coords = list[i]
    y, min_x, max_x = box_coords
    start_point = (4 * y - 40, 4 * min_x)
    end_point = (4 * y + 40, 4 * max_x + 120)
    image = cv2.rectangle(image, start_point, end_point, box_color, box_thickness)

cv2.imwrite("mask_ccoeff.jpg", mask)
cv2.imwrite("res15.jpg", image)

# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
#
# for meth in methods:
#     img = greek_ship.copy()
#     method = eval(meth)
#     res = cv2.matchTemplate(img, cropped_template, method)
#     res = cv2.normalize(src=res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     mask = np.zeros(res.shape)
#
#     if meth == 'cv2.TM_SQDIFF' or meth == 'cv2.TM_SQDIFF_NORMED':
#         mask[res < 50] = 255
#     else:
#         mask[res > 180] = 255
#     cv2.imwrite(meth[4:] + ".jpg", res)
#     cv2.imwrite("mask_"+meth[4:] + ".jpg", mask)
