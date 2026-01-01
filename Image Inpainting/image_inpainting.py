import random
import sys
import numpy as np
import cv2


def boundary_pixel_count(left, top, right=None, bottom=None):
    num_boundary_pixels = 0
    if left is not None:
        num_boundary_pixels += left.size
    if top is not None:
        num_boundary_pixels += top.size
    if right is not None:
        num_boundary_pixels += right.size
    if bottom is not None:
        num_boundary_pixels += bottom.size
    if left is not None and top is not None:
        top_left = top[:, 0:left.shape[1], :]
        num_boundary_pixels -= top_left.size
    if right is not None and top is not None:
        top_right = top[:, 0:right.shape[1], :]
        num_boundary_pixels -= top_right.size
    if left is not None and bottom is not None:
        bottom_left = bottom[:, 0:left.shape[1], :]
        num_boundary_pixels -= bottom_left.size
    if right is not None and bottom is not None:
        bottom_right = bottom[:, 0:right.shape[1], :]
        num_boundary_pixels -= bottom_right.size
    return num_boundary_pixels


def boundary_distance(num_boundary_pixels, left, top, right, bottom,
                      candidate_left, candidate_top, candidate_right, candidate_bottom):
    if candidate_left is None and candidate_top is None and candidate_right is None and candidate_bottom is None:
        return sys.maxsize
    sum_squared_error = 0
    if left is not None:
        sum_squared_error += np.sum(np.power(left - candidate_left, 2))
    if top is not None:
        sum_squared_error += np.sum(np.power(top - candidate_top, 2))
    if right is not None:
        sum_squared_error += np.sum(np.power(right - candidate_right, 2))
    if bottom is not None:
        sum_squared_error += np.sum(np.power(bottom - candidate_bottom, 2))

    # Subtract double-counted corner regions
    if left is not None and top is not None:
        top_left = top[:, 0:left.shape[1], :]
        sum_squared_error -= np.sum(np.power(top_left - candidate_top[:, 0:left.shape[1], :], 2))
    if right is not None and top is not None:
        top_right = top[:, 0:right.shape[1], :]
        sum_squared_error -= np.sum(np.power(top_right - candidate_top[:, 0:right.shape[1], :], 2))
    if left is not None and bottom is not None:
        bottom_left = bottom[:, 0:left.shape[1], :]
        sum_squared_error -= np.sum(np.power(bottom_left - candidate_bottom[:, 0:left.shape[1], :], 2))
    if right is not None and bottom is not None:
        bottom_right = bottom[:, 0:right.shape[1], :]
        sum_squared_error -= np.sum(np.power(bottom_right - candidate_bottom[:, 0:right.shape[1], :], 2))
    return np.sqrt(sum_squared_error / num_boundary_pixels)


def extract_candidate_boundaries(image, left, top, right, bottom, b_x, b_y):
    candidate_left, candidate_top, candidate_right, candidate_bottom = None, None, None, None
    if left is not None:
        left_h, left_w = left.shape[:2]
        candidate_left = image[b_x:b_x + left_h, b_y:b_y + left_w, :]
        if left.shape != candidate_left.shape:
            return None, None, None, None
    if top is not None:
        top_h, top_w = top.shape[:2]
        candidate_top = image[b_x:b_x + top_h, b_y:b_y + top_w, :]
        if top.shape != candidate_top.shape:
            return None, None, None, None
    if right is not None:
        right_h, right_w = right.shape[:2]
        right_x, right_y = b_x, b_y + top_w - right_w
        candidate_right = image[b_x:b_x+right_h, right_y:right_y+right_w, :]
        if right.shape != candidate_right.shape:
            return None, None, None, None
    if bottom is not None:
        bottom_h, bottom_w = bottom.shape[:2]
        bottom_x, bottom_y = b_x + left_h - bottom_h, b_y
        candidate_bottom = image[bottom_x:bottom_x+bottom_h, b_y:b_y+bottom_w, :]
        if bottom.shape != candidate_bottom.shape:
            return None, None, None, None
    return candidate_left, candidate_top, candidate_right, candidate_bottom


def get_rectangle_area(rect):
    x_start, y_start, x_end, y_end = rect
    return (x_end - x_start) * (y_end - y_start)


def split_texture_around_target(texture_x_start, texture_y_start, texture_x_end, texture_y_end,
                                target_x_start, target_y_start, target_x_end, target_y_end, patch_height, patch_width):
    top_left_rect = [texture_x_start,
                texture_y_start,
                target_x_end - patch_height,
                target_y_start - patch_width]
    bottom_left_rect = [texture_x_start,
                 target_y_start,
                 target_x_start - patch_height,
                 texture_y_end - patch_width]
    top_right = [target_x_end,
                   texture_y_start,
                   texture_x_end - patch_height,
                   target_y_end - patch_width]
    bottom_right = [target_x_start,
                  target_y_end,
                  texture_x_end - patch_height,
                  texture_y_end - patch_width]
    return top_left_rect, bottom_left_rect, top_right, bottom_right


def select_best_patch(num_candidates, image, patch_height, patch_width, left, top, right, bottom, candidate_points):
    num_boundary_pixels = boundary_pixel_count(left, top, right, bottom)
    min_distance = sys.maxsize
    best_candidates = {}
    for point in candidate_points:
        x, y = point
        candidate_left, candidate_top, candidate_right, candidate_bottom = extract_candidate_boundaries(image, left, top, right, bottom, x, y)
        d = boundary_distance(num_boundary_pixels, left, top, right, bottom, candidate_left, candidate_top, candidate_right, candidate_bottom)
        point = (x, y)
        if len(best_candidates) < num_candidates:
            best_candidates[d] = point

        elif min_distance > d:
            del best_candidates[min_distance]
            best_candidates[d] = point
        if min_distance > d:
            min_distance = d
    p_x, p_y = random.choice(list(best_candidates.values()))
    patch = image[p_x:p_x + patch_height, p_y:p_y + patch_width]
    return apply_min_cut_seam(patch, left, top, right, bottom)


def compute_vertical_min_cut(errors, keep_side):
    e_h, e_w = errors.shape[:2]
    dp_table = np.zeros((e_h, e_w, 3))
    path = np.zeros(errors.shape)
    cut_mask = np.zeros(errors.shape, dtype=bool)
    # last row initialize
    dp_table[e_h - 1, :, 0] = errors[e_h - 1, :]
    dp_table[e_h - 1, :, 1] = e_h - 1
    for j in range(e_w - 1, -1, -1):
        dp_table[e_h - 1, j, 2] = j
    # other rows
    for i in range(e_h - 2, -1, -1):
        for j in range(e_w - 1, -1, -1):
            e_ij = errors[i, j]
            min_cost_j = j
            min_cost = dp_table[i + 1, j, 0]
            if j - 1 >= 0:
                cost = dp_table[i + 1, j - 1, 0]
                if min_cost > cost:
                    min_cost = cost
                    min_cost_j = j - 1
            if j + 1 < e_w:
                cost = dp_table[i + 1, j + 1, 0]
                if min_cost > cost:
                    min_cost = cost
                    min_cost_j = j + 1
            min_cost += e_ij
            dp_table[i, j, :] = [min_cost, i + 1, min_cost_j]
    i, j = 0, int(np.min(dp_table[0, :, 2]))
    while True:
        path[i, j] = 1
        if keep_side == "left":
            cut_mask[i, :j] = True
        else:
            cut_mask[i, j:] = True
        if i == e_h - 1:
            break
        i = int(dp_table[i, j, 1])
        j = int(dp_table[i, j, 2])
    return path, cut_mask


def compute_horizontal_min_cut(errors, keep_side):
    e_h, e_w = errors.shape[:2]
    dp_table = np.zeros((e_h, e_w, 3))
    path = np.zeros(errors.shape)
    cut_mask = np.zeros(errors.shape, dtype=bool)
    # last column initialize
    dp_table[:, e_w - 1, 0] = errors[:, e_w - 1]
    dp_table[:, e_w - 1, 2] = e_w - 1
    for i in range(e_h - 1, -1, -1):
        dp_table[i, e_w - 1, 1] = i
    # other rows
    for j in range(e_w - 2, -1, -1):
        for i in range(e_h - 1, -1, -1):
            e_ij = errors[i, j]
            min_cost_i = i
            min_cost = dp_table[i, j + 1, 0]
            if i - 1 >= 0:
                cost = dp_table[i - 1, j + 1, 0]
                if min_cost > cost:
                    min_cost = cost
                    min_cost_i = i - 1
            if i + 1 < e_h:
                cost = dp_table[i + 1, j + 1, 0]
                if min_cost > cost:
                    min_cost = cost
                    min_cost_i = i + 1
            min_cost += e_ij
            dp_table[i, j, :] = [min_cost, min_cost_i, j + 1]
    i, j = int(np.min(dp_table[0, :, 1])), 0
    while True:
        path[i, j] = 1
        if keep_side == "top":
            cut_mask[:i, j] = True
        else:
            cut_mask[i:, j] = True
        if j == e_w - 1:
            break
        i = int(dp_table[i, j, 1])
        j = int(dp_table[i, j, 2])
    return path, cut_mask


def apply_min_cut_seam(patch, left, top, right, bottom):
    patch = patch.copy()
    p_h, p_w = patch.shape[:2]
    cut_mask = np.zeros_like(patch, dtype=bool)
    path = np.zeros((p_h, p_w))
    path_x_s, path_y_s, path_x_e, path_y_e = 0, 0, p_h, p_w
    boundary = np.zeros_like(patch)
    if left is not None:
        left_h, left_w = left.shape[:2]
        boundary[0:left_h, 0:left_w, :] = left
        left_color_error = np.power(left - patch[0:left_h, 0:left_w, :], 2)
        left_error = np.array(left_color_error[:, :, 0] + left_color_error[:, :, 1] + left_color_error[:, :, 2])
        left_path, left_cut_mask = compute_vertical_min_cut(left_error, "left")
        path[0:left_h, 0:left_w] += left_path
        for color in range(patch.shape[2]):
            cut_mask[0:left_h, 0:left_w, color] = left_cut_mask

    if top is not None:
        top_h, top_w = top.shape[:2]
        boundary[0:top_h, 0:top_w, :] = top
        top_color_error = np.power(top - patch[0:top_h, 0:top_w, :], 2)
        top_error = np.array(top_color_error[:, :, 0] + top_color_error[:, :, 1] + top_color_error[:, :, 2])
        top_path, top_cut_mask = compute_horizontal_min_cut(top_error, "top")
        path[0:top_h, 0:top_w] += top_path
        corners = np.array(np.where(path == 2))
        if corners.size != 0:
            path_x_s, path_y_s = corners[:, -1]
        for color in range(patch.shape[2]):
            cut_mask[0:top_h, 0:top_w, color] = top_cut_mask
    if right is not None:
        right_h, right_w = right.shape[:2]
        boundary[0:right_h, p_w-right_w:p_w, :] = right
        right_color_error = np.power(right - patch[0:right_h, p_w-right_w:p_w, :], 2)
        right_error = np.array(right_color_error[:, :, 0] + right_color_error[:, :, 1] + right_color_error[:, :, 2])
        right_path, right_cut_mask = compute_vertical_min_cut(right_error, "right")
        path[0:right_h, p_w-right_w:p_w] += right_path
        corners = np.array(np.where(path == 2))
        for color in range(patch.shape[2]):
            cut_mask[0:right_h, p_w-right_w:p_w, color] = right_cut_mask
    if bottom is not None:
        bottom_h, bottom_w = bottom.shape[:2]
        boundary[p_h-bottom_h:p_h, 0:bottom_w, :] = bottom
        bottom_color_error = np.power(bottom - patch[p_h - bottom_h:p_h, 0:bottom_w, :], 2)
        bottom_error = np.array(bottom_color_error[:, :, 0] + bottom_color_error[:, :, 1] + bottom_color_error[:, :, 2])
        bottom_path, bottom_cut_mask = compute_horizontal_min_cut(bottom_error, "bottom")
        path[p_h-bottom_h:p_h, 0:bottom_w] += bottom_path
        corners = np.where(path == 2)
        for color in range(patch.shape[2]):
            cut_mask[p_h-bottom_h:p_h, 0:bottom_w, color] = bottom_cut_mask
    np.copyto(patch, boundary, where=cut_mask)
    return patch


def get_texture_region(im_h, im_w, target_x_start, target_y_start, target_x_end, target_y_end, texture_margin_ratio):
    target_h, target_w = target_x_end - target_x_start, target_y_end - target_y_start
    if texture_margin_ratio < 0:
        texture_x_start, texture_y_start, texture_x_end, texture_y_end = 0, 0, im_h, im_w
    else:
        texture_x_start = target_x_start - texture_margin_ratio * target_h
        if texture_x_start < 0:
            texture_x_start = 0
        texture_y_start = target_y_start - texture_margin_ratio * target_w
        if texture_y_start < 0:
            texture_y_start = 0
        texture_x_end = target_x_end + texture_margin_ratio * target_h
        if texture_x_end > im_h:
            texture_x_end = im_h
        texture_y_end = target_y_end + texture_margin_ratio * target_w
        if texture_y_end > im_w:
            texture_y_end = im_w
    return texture_x_start, texture_y_start, texture_x_end, texture_y_end


def sample_points_in_rect(rect, num_samples):
    x_start, y_start, x_end, y_end = rect
    return [(np.random.randint(x_start, x_end), np.random.randint(y_start, y_end)) for _ in range(num_samples)]


def sample_random_points_from_regions(rect1, rect2, rect3, rect4, sampling_ratio):
    random_points = []
    random_points.extend(sample_points_in_rect(rect1, int(sampling_ratio * get_rectangle_area(rect1))))
    random_points.extend(sample_points_in_rect(rect2, int(sampling_ratio * get_rectangle_area(rect2))))
    random_points.extend(sample_points_in_rect(rect3, int(sampling_ratio * get_rectangle_area(rect3))))
    random_points.extend(sample_points_in_rect(rect4, int(sampling_ratio * get_rectangle_area(rect4))))
    return random_points


def inpaint_target_region(output_path, image, target_x_start, target_y_start, target_x_end, target_y_end,
                          patch_size_ratio, num_candidates, overlap_ratio, sampling_ratio, texture_margin_ratio=-1):
    im_h, im_w = image.shape[:2]
    target_h, target_w = target_x_end - target_x_start, target_y_end - target_y_start
    image[target_x_start:target_x_end, target_y_start:target_y_end, :] = np.zeros((target_h, target_w, 3))
    texture_x_start, texture_y_start, texture_x_end, texture_y_end = get_texture_region(im_h, im_w, target_x_start, target_y_start, target_x_end, target_y_end, texture_margin_ratio)
    patch_size = int(patch_size_ratio * min(target_h, target_w))
    patch_h, patch_w = patch_size, patch_size
    top_left_rect, bottom_left_rect, top_right_rect, bottom_right_rect = split_texture_around_target(texture_x_start, texture_y_start, texture_x_end, texture_y_end, target_x_start, target_y_start, target_x_end, target_y_end, patch_h, patch_w)
    overlap_size = int(patch_size * overlap_ratio)
    non_overlap_size = patch_size - overlap_size

    for x in range(target_x_start - overlap_size, target_x_end - overlap_size, non_overlap_size):
        remaining_height = target_x_end + overlap_size - x
        if remaining_height < patch_size:
            patch_h = remaining_height
        patch_w = patch_size
        for y in range(target_y_start - overlap_size, target_y_end - overlap_size, non_overlap_size):
            remaining_width = target_y_end + overlap_size - y
            if remaining_width < patch_size:
                patch_w = remaining_width
            patch_h, patch_w = patch_size, patch_size
            left, top, right, bottom = None, None, None, None
            left = image[x:x + patch_h, y:y + overlap_size, :]
            top = image[x:x + overlap_size, y:y + patch_w, :]
            if remaining_width < patch_size:
                right = image[x:x + patch_h, y + patch_w - overlap_size:y + patch_w, :]
            if remaining_height < patch_size:
                bottom = image[x + patch_h - overlap_size:x + patch_h, y:y + patch_w, :]
            candidate_patch_points = sample_random_points_from_regions(top_left_rect, bottom_left_rect, top_right_rect, bottom_right_rect, sampling_ratio)
            patch = select_best_patch(num_candidates, image, patch_h, patch_w, left, top, right, bottom, candidate_patch_points)
            image[x:x+patch_h, y:y+patch_w, :] = patch
            cv2.imwrite(output_path, image)
    return image


img = cv2.imread("im03.jpg", cv2.IMREAD_UNCHANGED)
output_path = "res15.jpg"
patch_size_ratio = 0.25
num_candidates = 1
overlap_ratio = 0.2
sampling_ratio = 0.05
texture_margin_ratio = 1
target1_x_start, target1_y_start, target1_x_end, target1_y_end = 62, 323, 173, 544
img = inpaint_target_region(output_path, img, target1_x_start, target1_y_start, target1_x_end, target1_y_end, patch_size_ratio, num_candidates, overlap_ratio, sampling_ratio, texture_margin_ratio)
target2_x_start, target2_y_start, target2_x_end, target2_y_end = 749, 829, 933, 972
img = inpaint_target_region(output_path, img, target2_x_start, target2_y_start, target2_x_end, target2_y_end, patch_size_ratio, num_candidates, overlap_ratio, sampling_ratio, texture_margin_ratio)
target3_x_start, target3_y_start, target3_x_end, target3_y_end = 612, 1133, 711, 1228
img = inpaint_target_region(output_path, img, target3_x_start, target3_y_start, target3_x_end, target3_y_end, patch_size_ratio, num_candidates, overlap_ratio, sampling_ratio, texture_margin_ratio)
cv2.imwrite(output_path, img)

img = cv2.imread("im04.jpg", cv2.IMREAD_UNCHANGED)
output_path = "res16.jpg"
patch_size_ratio = 0.7
num_candidates = 1
overlap_ratio = 0.1
sampling_ratio = 0.05
texture_margin_ratio = -1
target_x_start, target_y_start, target_x_end, target_y_end = 695, 743, 1173, 949
img = inpaint_target_region(output_path, img, target_x_start, target_y_start, target_x_end, target_y_end, patch_size_ratio, num_candidates, overlap_ratio, sampling_ratio)
cv2.imwrite(output_path, img)




