import time
import cv2
import numpy as np


def downsample_by_half(image):
    new_height = image.shape[0] // 2
    new_width = image.shape[1] // 2
    new_img = np.zeros((new_height, new_width))
    for i in range(new_height):
        row = i * 2
        for j in range(new_width):
            col = j * 2
            new_img[i][j] = image[row][col]
    return new_img


def build_gaussian_pyramid(image, pyramid_levels):
    pyramid = [image]
    for i in range(1, pyramid_levels):
        pyramid.append(downsample_by_half(pyramid[i - 1]))
    return pyramid


def reference_channel_bounds(limit, shift):
    start = max(0, -shift)
    end = min(limit, limit - shift)
    return start, end


def other_channel_bounds(limit, shift):
    start = max(0, shift)
    end = min(limit, limit + shift)
    return start, end


def compute_ssd(height, width, reference_channel, other_channel, shift_x, shift_y):
    # Compute the sum of squared differences between two channels for a given shift
    x_start, x_end = reference_channel_bounds(height, shift_x)
    y_start, y_end = reference_channel_bounds(width, shift_y)
    overlap_area = (x_end - x_start) * (y_end - y_start)
    if overlap_area / (height * width) < MIN_OVERLAP_RATIO:
        return float('inf')
    diff = reference_channel[x_start:x_end, y_start:y_end] - \
           other_channel[x_start + shift_x:x_end + shift_x, y_start + shift_y:y_end + shift_y]
    sd = np.multiply(diff, diff)
    return np.sum(sd)


def compute_sad(height, width, reference_channel, other_channel, shift_x, shift_y):
    # Compute the sum of absolute differences between two channels for a given shift
    x_start, x_end = reference_channel_bounds(height, shift_x)
    y_start, y_end = reference_channel_bounds(width, shift_y)
    overlap_area = (x_end - x_start) * (y_end - y_start)
    if overlap_area / (height * width) < MIN_OVERLAP_RATIO:
        return float('inf')
    diff = reference_channel[x_start:x_end, y_start:y_end] - \
           other_channel[x_start + shift_x:x_end + shift_x, y_start + shift_y:y_end + shift_y]
    sd = np.abs(diff)
    return np.sum(sd)


def find_best_shift(reference_channel, other_channel, x_start, x_end, y_start, y_end):
    # Brute-force search for the best x/y shift between two channels using SSD
    min_ssd = float('inf')
    h, w = reference_channel.shape[:2]
    best_x, best_y = 0, 0
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            ssd = compute_ssd(h, w, reference_channel, other_channel, x, y)
            if min_ssd > ssd:
                min_ssd = ssd
                best_x, best_y = x, y

    return best_x, best_y


def pyramid_best_shifts(pyramid_levels, reference_channel_pyramid, other_channel_pyramid):
    offsets = np.zeros((pyramid_levels, 2), dtype=int)
    match_x, match_y = 0, 0
    # search for match in range [match - s, match + s +1]
    s = max(reference_channel_pyramid[pyramid_levels - 1].shape[0] // 10, reference_channel_pyramid[pyramid_levels - 1].shape[1] // 10)

    for l in reversed(range(pyramid_levels)):
        main_channel = reference_channel_pyramid[l]
        other_channel = other_channel_pyramid[l]
        h, w = main_channel.shape[:2]

        x_start, x_end = match_x - s, match_x + s + 1
        y_start, y_end = match_y - s, match_y + s + 1
        # remove black border
        border_h = h // 10
        border_w = w // 10
        match_x, match_y = find_best_shift(main_channel[border_h:h - border_h, border_w:w - border_w],
                                           other_channel[border_h:h - border_h, border_w:w - border_w], x_start, x_end,
                                           y_start, y_end)

        offsets[l, :] = [match_x, match_y]

        match_x *= 2
        match_y *= 2
        s = pow(2, l + 1)

    return offsets


def detect_edge(channel, region, s_h, s_w):
    edge_filter = [1, 0, -1]
    channel = channel.astype(np.float32)
    h, w = channel.shape[:2]
    if region == "left":
        filtered = channel[0:s_h - 2, :] * edge_filter[0] + channel[2:s_h, :] * edge_filter[2]
        mean = np.abs(filtered.mean(axis=1))
        edge_indices = np.argwhere(mean > EDGE_THRESHOLD)
        edge = 1 + np.max(edge_indices) if edge_indices.size != 0  else 0
    elif region == "right":
        filtered = channel[h - s_h: h - 2, :] * edge_filter[0] + channel[h - s_h + 2: h, :] * edge_filter[2]
        mean = np.abs(filtered.mean(axis=1))
        edge_indices = np.argwhere(mean > EDGE_THRESHOLD)
        edge = h - s_h - 1 + np.min(edge_indices) if edge_indices.size != 0  else h
    elif region == "up":
        filtered = channel[:, 0:s_w - 2] * edge_filter[0] + channel[:, 2:s_w] * edge_filter[2]
        mean = np.abs(filtered.mean(axis=0))
        edge_indices = np.argwhere(mean > EDGE_THRESHOLD)
        edge = 1 + np.max(edge_indices) if edge_indices.size != 0  else 0
    elif region == "down":
        filtered = channel[:, w - s_w: w - 2] * edge_filter[0] + channel[:, w - s_w + 2: w] * edge_filter[2]
        mean = np.abs(filtered.mean(axis=0))
        edge_indices = np.argwhere(mean > EDGE_THRESHOLD)
        edge = w - s_w - 1 + np.min(edge_indices) if edge_indices.size != 0  else w
    return edge


def compute_border(channel, l):
    # Compute content borders of a single channel to ignore black edges
    h, w = channel.shape[:2]
    if l >= 3:
        s_h = h // 10
        s_w = w // 10
    else:
        s_h = h // 20
        s_w = w // 20

    left_edge = detect_edge(channel, "left", s_h, s_w)
    right_edge = detect_edge(channel, "right", s_h, s_w)
    up_edge = detect_edge(channel, "up", s_h, s_w)
    down_edge = detect_edge(channel, "down", s_h, s_w)
    return left_edge, right_edge, up_edge, down_edge


def place_shifted_channel(merged, shifted_channel, c, shift_x, shift_y):
    h, w = shifted_channel.shape
    ref_x_start, ref_x_end = reference_channel_bounds(h, shift_x)
    ref_y_start, ref_y_end = reference_channel_bounds(w, shift_y)
    x_start, x_end = other_channel_bounds(h, shift_x)
    y_start, y_end = other_channel_bounds(w, shift_y)
    merged[ref_x_start:ref_x_end, ref_y_start:ref_y_end, c] = shifted_channel[x_start:x_end, y_start:y_end]
    return ref_x_start, ref_x_end, ref_y_start, ref_y_end


def merge_channels(blue_channel, green_channel, red_channel, blue_shift_x, blue_shift_y, red_shift_x, red_shift_y, crop=True):
    h, w = blue_channel.shape[:2]
    merged = np.zeros((h, w, 3))
    merged[:, :, 1] = green_channel
    b_x_start, b_x_end, b_y_start, b_y_end = place_shifted_channel(merged, blue_channel, 0, blue_shift_x, blue_shift_y)
    r_x_start, r_x_end, r_y_start, r_y_end = place_shifted_channel(merged, red_channel, 2, red_shift_x, red_shift_y)
    if crop:
        max_x_s = max(b_x_start, r_x_start)
        min_x_e = min(b_x_end, r_x_end)
        max_y_s = max(b_y_start, r_y_start)
        min_y_e = min(b_y_end, r_y_end)
        merged = merged[max_x_s:min_x_e, max_y_s:min_y_e]
    return merged


def crop_to_content(level, cropped_blue, cropped_green, cropped_red, b_x, b_y, r_x, r_y):
    blue_sx, blue_ex, blue_sy, blue_y_e = compute_border(cropped_blue, level)
    green_sx, green_ex, green_sy, green_ey = compute_border(cropped_green, level)
    red_sx, red_ex, red_sy, red_ey = compute_border(cropped_red, level)
    x_start = max(blue_sx, green_sx, red_sx)
    x_end = min(blue_ex, green_ex, red_ex)
    y_start = max(blue_sy, green_sy, red_sy)
    y_end = min(blue_y_e, green_ey, red_ey)
    cropped_blue = cropped_blue[x_start:x_end, y_start:y_end]
    cropped_green = cropped_green[x_start:x_end, y_start:y_end]
    cropped_red = cropped_red[x_start:x_end, y_start:y_end]
    cropped_image = merge_channels(cropped_blue, cropped_green, cropped_red, b_x, b_y, r_x, r_y, crop=True)
    return cropped_image


def align_and_crop_image(image_path, pyramid_levels, image_name, save_cropped=True):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[0] // 3, image.shape[1]

    blue_channel = image[0:h, :]
    green_channel = image[h:2 * h, :]
    red_channel = image[2 * h:3 * h, :]

    blue_pyramid = build_gaussian_pyramid(blue_channel, pyramid_levels)
    green_pyramid = build_gaussian_pyramid(green_channel, pyramid_levels)
    red_pyramid = build_gaussian_pyramid(red_channel, pyramid_levels)

    blue_offsets = pyramid_best_shifts(pyramid_levels, green_pyramid, blue_pyramid)
    red_offsets = pyramid_best_shifts(pyramid_levels, green_pyramid, red_pyramid)

    final_image = None
    for l in range(pyramid_levels):
        blue_x, blue_y = blue_offsets[l, :]
        red_x, red_y = red_offsets[l, :]
        image = merge_channels(blue_pyramid[l], green_pyramid[l], red_pyramid[l], blue_x, blue_y, red_x, red_y) // 256
        cropped_image = crop_to_content(l, blue_pyramid[l], green_pyramid[l], red_pyramid[l], blue_x, blue_y, red_x, red_y) // 256
        if l == 0:
            final_image = cropped_image
        cv2.imwrite(image_name + "_" + str(l) + ".jpg", image)
        if save_cropped:
            cv2.imwrite("cropped_" + image_name + "_" + str(l) + ".jpg", cropped_image)
    return final_image


EDGE_FILTER = [1, 0, -1]
MIN_OVERLAP_RATIO = 0.95
EDGE_THRESHOLD = 500
LEVELS = 6


image_path = "master-pnp-prok-01800-01833a.tif"
result = align_and_crop_image(image_path, LEVELS, "Mosque", save_cropped=False)
cv2.imwrite("res04_Mosque.jpg", result)


