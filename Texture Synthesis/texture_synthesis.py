import math
import multiprocessing
from joblib import Parallel, delayed
import random
import sys
import time
import cv2
import numpy as np


def downsample_by_2_gray(image, row_offset=0, col_offset=0):
    shape = image.shape
    im_h, im_w = shape[:2]
    half_h = math.ceil((im_h - row_offset) / 2)
    half_w = math.ceil((im_w - col_offset) / 2)
    half_img = np.zeros((half_h, half_w))
    for i in range(0, im_h):
        im_i = row_offset + 2 * i
        if im_i < im_h:
            for j in range(0, im_w):
                im_j = col_offset + 2 * j
                if im_j < im_w:
                    half_img[i, j] = image[im_i, im_j]
    return half_img


def downsample_by_2(image, row_offset=0, col_offset=0):
    shape = image.shape
    if len(shape) == 2:
        return downsample_by_2_gray(image, row_offset, col_offset)
    im_h, im_w = shape[:2]
    half_h = math.ceil((im_h - row_offset) / 2)
    half_w = math.ceil((im_w - col_offset) / 2)
    half_img = np.zeros((half_h, half_w, image.shape[2]))
    for i in range(0, im_h):
        im_i = row_offset + 2 * i
        if im_i < im_h:
            for j in range(0, im_w):
                im_j = col_offset + 2 * j
                if im_j < im_w:
                    half_img[i, j, :] = image[im_i, im_j, :]
    return half_img


def quadtree_node_count(pyramid_level):
    num_nodes = 0
    for l in range(pyramid_level):
        num_nodes += np.power(4, l)
    return num_nodes


def QTP(image, pyramid_level):
    num_nodes = quadtree_node_count(pyramid_level)
    pyramid_nodes = []
    pyramid_nodes.append(image)
    parent_index = 0
    child_offset_index = 0
    for i in range(1, num_nodes):
        row_offset, col_offset = child_offset_index // 2, child_offset_index % 2
        pyramid_nodes.append(downsample_by_2(pyramid_nodes[parent_index], row_offset, col_offset))
        child_offset_index += 1
        if child_offset_index > 3:
            child_offset_index = 0
            parent_index += 1
    return pyramid_nodes


def QTP_boundary(pyramid_level, left, top, right, bottom):
    left_qtp, top_qtp, right_qtp, bottom_qtp = None, None, None, None
    if left is not None:
        left_qtp = QTP(left, pyramid_level)
    if top is not None:
        top_qtp = QTP(top, pyramid_level)
    if right is not None:
        right_qtp = QTP(right, pyramid_level)
    if bottom is not None:
        bottom_qtp = QTP(bottom, pyramid_level)
    return left_qtp, top_qtp, right_qtp, bottom_qtp


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


def boundary_distance_threshold(distance_tolerance, num_boundary_pixels, left, top, right=None, bottom=None):
    sum_squared_pixels = 0
    if left is not None:
        sum_squared_pixels += np.sum(np.power(left, 2))
    if top is not None:
        sum_squared_pixels += np.sum(np.power(top, 2))
    if right is not None:
        sum_squared_pixels += np.sum(np.power(right, 2))
    if bottom is not None:
        sum_squared_pixels += np.sum(np.power(bottom, 2))

    # Remove double-counted corners
    if left is not None and top is not None:
        top_left = top[:, 0:left.shape[1], :]
        sum_squared_pixels -= np.sum(np.power(top_left, 2))
    if right is not None and top is not None:
        top_right = top[:, 0:right.shape[1], :]
        sum_squared_pixels -= np.sum(np.power(top_right, 2))
    if left is not None and bottom is not None:
        bottom_left = bottom[:, 0:left.shape[1], :]
        sum_squared_pixels -= np.sum(np.power(bottom_left, 2))
    if right is not None and bottom is not None:
        bottom_right = bottom[:, 0:right.shape[1], :]
        sum_squared_pixels -= np.sum(np.power(bottom_right, 2))
    return distance_tolerance * np.sqrt(sum_squared_pixels / num_boundary_pixels)


def boundary_distance(num_boundary_pixels, left, top, right, bottom, candidate_left, candidate_top, candidate_right, candidate_bottom):
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


def extract_candidate_boundaries(texture, left, top, right, bottom, x, y):
    candidate_left, candidate_top, candidate_right, candidate_bottom = None, None, None, None
    if left is not None:
        left_h, left_w = left.shape[:2]
        candidate_left = texture[x:x+left_h, y:y+left_w, :]
        if left.shape != candidate_left.shape:
            return None, None, None, None
    if top is not None:
        top_h, top_w = top.shape[:2]
        candidate_top = texture[x:x+top_h, y:y+top_w, :]
        if top.shape != candidate_top.shape:
            return None, None, None, None
    if right is not None:
        right_h, right_w = right.shape[:2]
        right_x, right_y = x, x + top_w - right_w
        candidate_right = texture[x:x + right_h, right_y:right_y + right_w, :]
        if right.shape != candidate_right.shape:
            return None, None, None, None
    if bottom is not None:
        bottom_h, bottom_w = bottom.shape[:2]
        bottom_x, bottom_y = x + left_h - bottom_h, y
        candidate_bottom = texture[bottom_x:bottom_x + bottom_h, y:y + bottom_w, :]
        if bottom.shape != candidate_bottom.shape:
            return None, None, None, None
    return candidate_left, candidate_top, candidate_right, candidate_bottom


def random_patch(texture, p_size):
    h, w = texture.shape[:2]
    x = np.random.randint(0, h - p_size)
    y = np.random.randint(0, w - p_size)
    return texture[x:x + p_size, y:y + p_size, :]


def valid_patch_search_region(t_h, t_w, left, top, right, bottom):
    min_x, max_x, mid_y, max_y = 0, t_h, 0, t_w
    has_upper_left_constraint = False
    has_lower_right_constraint = False
    if left is not None:
        p_size = left.shape[0]
        max_x -= (p_size + 2)
        max_y -= (p_size + 2)
        has_upper_left_constraint = True
    if top is not None:
        if has_upper_left_constraint is False:
            p_size = top.shape[1]
            max_x -= (p_size + 2)
            max_y -= (p_size + 2)
    if right is not None:
        p_size = right.shape[0]
        min_x = p_size + 2
        mid_y = p_size + 2
        has_lower_right_constraint = True
    if bottom is not None:
        if has_lower_right_constraint is False:
            p_size = top.shape[1]
            min_x = p_size + 2
            mid_y = p_size + 2
    return min_x, max_x, mid_y, max_y


def compute_vertical_min_cut(errors, keep_side):
    e_h, e_w = errors.shape[:2]
    dp_table = np.zeros((e_h, e_w, 3))
    path = np.zeros(errors.shape, dtype=bool)
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
        path[i, j] = True
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
    path = np.zeros(errors.shape, dtype=bool)
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
        path[i, j] = True
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
    cut_mask = np.zeros_like(patch, dtype=bool)
    boundary = np.zeros_like(patch)
    if left is not None:
        left_h, left_w = left.shape[:2]
        boundary[0:left_h, 0:left_w, :] = left
        left_color_error = np.power(left - patch[0:left_h, 0:left_w, :], 2)
        left_error = np.array(left_color_error[:, :, 0] + left_color_error[:, :, 1] + left_color_error[:, :, 2])
        left_path, left_cut_mask = compute_vertical_min_cut(left_error, "left")
        for color in range(patch.shape[2]):
            cut_mask[0:left_h, 0:left_w, color] = left_cut_mask

    if top is not None:
        top_h, top_w = top.shape[:2]
        boundary[0:top_h, 0:top_w, :] = top
        top_color_error = np.power(top - patch[0:top_h, 0:top_w, :], 2)
        top_error = np.array(top_color_error[:, :, 0] + top_color_error[:, :, 1] + top_color_error[:, :, 2])
        top_path, top_cut_mask = compute_horizontal_min_cut(top_error, "top")
        for color in range(patch.shape[2]):
            cut_mask[0:top_h, 0:top_w, color] = top_cut_mask
    # TODO right, down
    np.copyto(patch, boundary, where=cut_mask)
    return patch


def select_patch_candidates(is_root_node, num_patch_candidates, distance_tolerance, texture,
                            left, top, right, bottom, num_random_point, inherited_points=None):
    num_boundary_pixels = boundary_pixel_count(left, top, right, bottom)
    max_boundary_distance = boundary_distance_threshold(distance_tolerance, num_boundary_pixels, left, top, right, bottom)
    t_h, t_w = texture.shape[:2]
    x_start, x_end, y_start, y_end = valid_patch_search_region(t_h, t_w, left, top, right, bottom)
    min_distance = sys.maxsize
    best_candidate_point = 0
    valid_patch_points = []
    point_dict = {}
    if inherited_points is None:
        random_points = [(np.random.randint(x_start, x_end), np.random.randint(y_start, y_end)) for _ in range(num_random_point)]
        for point in random_points:
            x, y = point
            candidate_left, candidate_top, candidate_right, candidate_bottom = extract_candidate_boundaries(texture, left, top, right, bottom, x, y)
            d = boundary_distance(num_boundary_pixels, left, top, right, bottom, candidate_left, candidate_top, candidate_right, candidate_bottom)
            point = (2 * x, 2 * y)
            if is_root_node:
                point = (x, y)
                if len(point_dict) < num_patch_candidates:
                    point_dict[d] = point
                elif min_distance > d:
                    del point_dict[min_distance]
                    point_dict[d] = point
            if d < max_boundary_distance:
                valid_patch_points.append(point)
            if min_distance > d:
                min_distance = d
                best_candidate_point = point
    else:
        for point in inherited_points:
            x, y = point
            candidate_left, candidate_top, candidate_right, candidate_bottom = extract_candidate_boundaries(texture, left, top, right, bottom, x, y)
            d = boundary_distance(num_boundary_pixels, left, top, right, bottom, candidate_left, candidate_top, candidate_right, candidate_bottom)
            point = (2 * x, 2 * y)
            if is_root_node:
                point = (x, y)
                if len(point_dict) < num_patch_candidates:
                    point_dict[d] = point
                elif min_distance > d:
                    del point_dict[min_distance]
                    point_dict[d] = point
            if d < max_boundary_distance:
                valid_patch_points.append(point)
            if min_distance > d:
                min_distance = d
                best_candidate_point = point
    if is_root_node:
        return list(point_dict.values())
    if not valid_patch_points:
        valid_patch_points.append(best_candidate_point)
    return valid_patch_points


def select_patch_multiresolution(num_patch_candidates, distance_tolerance, pyramid_level, patch_size,
                                 random_sample_ratio, texture_qtp, left, top, right, bottom):
    left_qtp, top_qtp, right_qtp, bottom_qtp = QTP_boundary(pyramid_level, left, top, right, bottom)
    num_quadtree_nodes = quadtree_node_count(pyramid_level)
    patch_candidates_per_node = []
    for i in range(num_quadtree_nodes):
        patch_candidates_per_node.append([])
    for i in range(num_quadtree_nodes - 1, -1, -1):
        node_texture = texture_qtp[i]
        node_left, node_top, node_right, node_bottom = None, None, None, None
        if left is not None:
            node_left = left_qtp[i]
        if top is not None:
            node_top = top_qtp[i]
        if right is not None:
            node_right = right_qtp[i]
        if bottom is not None:
            node_bottom = bottom_qtp[i]

        inherited_points = None
        if 4 * i + 3 < num_quadtree_nodes :
            inherited_points = []
            for l in range(1, 5):
                inherited_points.extend(patch_candidates_per_node[4 * i + l])
        num_random_point = int(random_sample_ratio * node_texture.size)
        patch_candidates_per_node[i] = select_patch_candidates(i == 0, num_patch_candidates, distance_tolerance, node_texture,
                                                               node_left, node_top, node_right, node_bottom, num_random_point, inherited_points)
    full_texture = texture_qtp[0]
    p_x, p_y = random.choice(patch_candidates_per_node[0])
    patch = full_texture[p_x:p_x + patch_size, p_y:p_y + patch_size]
    return apply_min_cut_seam(patch, left, top, right, bottom)


def synthesize_texture(output_path, target_height, target_width, texture, patch_size_ratio,
                       num_patch_candidates, distance_tolerance, pyramid_level, overlap_ratio, random_sample_ratio):
    t_h, t_w = texture.shape[:2]
    patch_size = int(patch_size_ratio * min(t_h, t_w))
    overlap_size = int(patch_size * overlap_ratio)
    non_overlap_size = patch_size - overlap_size
    num_patches_vert, num_patches_horiz = int(target_height // non_overlap_size) + 1, int(target_width // non_overlap_size) + 1
    padded_output_height, padded_output_width = num_patches_vert * non_overlap_size + overlap_size, num_patches_horiz * non_overlap_size + overlap_size
    synthesized_texture = np.zeros((padded_output_height, padded_output_width, texture.shape[2]))
    texture_qtp = QTP(texture, pyramid_level)
    for patch_x in range(0, padded_output_height - overlap_size, non_overlap_size):
        for patch_y in range(0, padded_output_width - overlap_size, non_overlap_size):
            left, top, right, bottom = None, None, None, None
            print(patch_x, ",", patch_y)
            if patch_x == 0 and patch_y == 0:
                patch = random_patch(texture, patch_size)
            else:
                if patch_x == 0:
                    left = synthesized_texture[0:patch_size, patch_y:patch_y + overlap_size, :]
                elif patch_y == 0:
                    top = synthesized_texture[patch_x:patch_x + overlap_size, 0:patch_size, :]
                else:
                    left = synthesized_texture[patch_x:patch_x + patch_size, patch_y:patch_y + overlap_size, :]
                    top = synthesized_texture[patch_x:patch_x + overlap_size, patch_y:patch_y + patch_size, :]
                patch = select_patch_multiresolution(num_patch_candidates, distance_tolerance, pyramid_level, patch_size, random_sample_ratio, texture_qtp, left, top, right, bottom)
            synthesized_texture[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size, :] = patch
            cv2.imwrite(output_path, synthesized_texture[0:target_height, 0:target_width, :])
    return synthesized_texture[0:target_height, 0:target_width, :]


def synthesize_patch_worker(patch_index_in_diagonal):
    left, top, right, bottom = None, None, None, None
    patch_x, patch_y = (diagonal_index - patch_index_in_diagonal) * non_overlap_size, patch_index_in_diagonal * non_overlap_size
    if patch_x == 0 and patch_y == 0:
        first_patch = random_patch(texture_qtp[0], patch_size)
        synthesized_texture[0:patch_size, 0:patch_size, :] = first_patch
    else:
        if patch_x == 0:
            left = synthesized_texture[0:patch_size, patch_y:patch_y + overlap_size, :]
        elif patch_y == 0:
            top = synthesized_texture[patch_x:patch_x + overlap_size, 0:patch_size, :]
        else:
            left = synthesized_texture[patch_x:patch_x + patch_size, patch_y:patch_y + overlap_size, :]
            top = synthesized_texture[patch_x:patch_x + overlap_size, patch_y:patch_y + patch_size, :]
        patch = select_patch_multiresolution(num_candidates, distance_tolerance, pyramid_level, patch_size, random_sample_ratio,
                                             texture_qtp, left, top, right, bottom)
        synthesized_texture[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size, :] = patch
    cv2.imwrite(output_path, synthesized_texture[0:output_size, 0:output_size, :])
    return


if __name__ == "__main__":
    source_texture = cv2.imread("texture01.jpg", cv2.IMREAD_UNCHANGED)
    t_h, t_w = source_texture.shape[:2]

    output_path = "res11.jpg"
    output_size = 2500
    patch_size_ratio = 0.25
    distance_tolerance = 0.35
    pyramid_level = 1
    num_candidates = 1
    random_sample_ratio = 0.002
    overlap_ratio = 0.2

    patch_size = int(patch_size_ratio * min(t_h, t_w))
    overlap_size = int(patch_size * overlap_ratio)
    non_overlap_size = patch_size - overlap_size
    texture_qtp = QTP(source_texture, pyramid_level)
    num_patches_per_axis = int(output_size // non_overlap_size) + 1
    padded_output_size = num_patches_per_axis * non_overlap_size + overlap_size

    synthesized_texture = np.zeros((padded_output_size, padded_output_size, source_texture.shape[2]))
    num_diagonals = 2 * num_patches_per_axis - 1
    diagonal_start_col = 0
    num_cores = multiprocessing.cpu_count()
    for diagonal_index in range(0, num_diagonals):
        if diagonal_index >= num_patches_per_axis:
            diagonal_start_col += 1
        Parallel(n_jobs=num_cores, require='sharedmem')(
            delayed(synthesize_patch_worker)(j) for j in range(diagonal_start_col, min(diagonal_index + 1, num_patches_per_axis)))


