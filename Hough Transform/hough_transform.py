import sys
import cv2
import numpy as np


def compute_hough_accumulator(edge_image, num_theta=180, num_rho=180):
    h, w = edge_image.shape
    half_h, half_w = h // 2, w // 2
    d = np.sqrt(h ** 2 + w ** 2)
    theta_step = 180 / num_theta
    rho_step = (2 * d) / num_rho
    theta_values = np.arange(0, 180, theta_step)
    rho_values = np.arange(-d, d, rho_step)
    cos_theta = np.cos(np.deg2rad(theta_values))
    sin_theta = np.sin(np.deg2rad(theta_values))
    accumulator = np.zeros((rho_values.size, theta_values.size))
    edge_pixels = np.array(np.where(edge_image > 0))
    for i in range(edge_pixels.shape[1]):
        x, y = edge_pixels[:, i]
        # Shift origin to the center of the image
        new_x, new_y = x - half_h, y - half_w
        vote_mask = np.zeros((rho_values.size, theta_values.size), dtype=bool)
        rho = new_x * sin_theta + new_y * cos_theta
        for thetas_idx in range(theta_values.size):
            closest_rho_idx = np.argmin(np.abs(rho[thetas_idx] - rho_values))
            vote_mask[closest_rho_idx, thetas_idx] = True
        accumulator[vote_mask] += 1
    return accumulator, rho_values, theta_values


def extract_hough_lines(accumulator, rhos, thetas, threshold=200):
    # rho = x * cos(theta) + y * sin(theta) <-> y = mx + d
    line_indices = np.array(np.where(accumulator > threshold))
    lines = []
    for i in range(line_indices.shape[1]):
        rho_idx, theta_idx = line_indices[:, i]
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        cos_theta = np.cos(np.deg2rad(theta))
        sin_theta = np.sin(np.deg2rad(theta))
        if sin_theta == 0:
            m = sys.maxsize
            d = sys.maxsize
        else:
            m = - cos_theta/sin_theta
            d = rho/sin_theta
        lines.append([rho, theta, m, d])
    return lines


def filter_chessboard_lines(gray_image, lines, chess_x_s, chess_x_e, chess_y_s, chess_y_e):
    unique_lines = merge_similar_lines(lines)
    im_h, im_w = gray_image.shape[:2]
    half_im_h, half_im_w = im_h // 2, im_w // 2
    chess_lines = []
    for line in unique_lines:
        # Chessboard lines have a high intensity difference across them
        # (because squares alternate between white=255 and black=0)
        rho, theta = line
        intensity_diff = 0
        for i in range(1, 5):
            r1 = rho + i
            r2 = rho - i
            x1_line, y1_line = sample_line_points(im_h, im_w, half_im_h, half_im_w, r1, theta)
            x2_line, y2_line = sample_line_points(im_h, im_w, half_im_h, half_im_w, r2, theta)
            for i in range(min(x1_line.size, x2_line.size)):
                x1, y1 = int(x1_line[i]), int(y1_line[i])
                x2, y2 = int(x2_line[i]), int(y2_line[i])
                if chess_x_s <= x1 < chess_x_e and chess_y_s <= y1 < chess_y_e and\
                        chess_x_s <= x2 < chess_x_e and chess_y_s <= y2 < chess_y_e:
                    intensity_diff += abs(int(gray_image[x1, y1] - gray_image[x2, y2]))
        if intensity_diff > 100000:
            chess_lines.append(line)
    return chess_lines


def merge_similar_lines(lines, m_tolerance=1, d_tolerance=20):
    unique_lines = []
    while True:
        if len(lines) == 0:
            break
        rho, theta, n = cluster_lines_around_specific_line(lines, 0, m_tolerance, d_tolerance)
        unique_lines.append([rho, theta])
    return unique_lines


def cluster_lines_around_specific_line(lines, line_idx, m_tolerance, d_tolerance):
    rho, theta, m, d = lines[line_idx]
    clustered_lines = []
    avg_rho, avg_theta, num_lines_merged = 0, 0, 0
    lines_num = len(lines)
    index_flag = False
    for i in range(lines_num):
        if index_flag:
            i -= 1
            index_flag = False
        if i >= lines_num:
            break
        candidate_rho, candidate_theta, candidate_m, candidate_d = lines[i]
        if len(clustered_lines) == 0:
            if np.abs(m - candidate_m) < m_tolerance and np.abs(d - candidate_d) < d_tolerance:
                del lines[i]
                avg_rho += candidate_rho
                avg_theta += candidate_theta
                clustered_lines.append([candidate_m, candidate_d])
                num_lines_merged += 1
                index_flag = True
                lines_num -= 1
        else:
            for j in range(len(clustered_lines)):
                m, d = clustered_lines[j]
                if np.abs(m - candidate_m) < m_tolerance and np.abs(d - candidate_d) < d_tolerance:
                    del lines[i]
                    avg_rho += candidate_rho
                    avg_theta += candidate_theta
                    clustered_lines.append([candidate_m, candidate_d])
                    num_lines_merged += 1
                    index_flag = True
                    lines_num -= 1
                    break
    return avg_rho/num_lines_merged, avg_theta/num_lines_merged, num_lines_merged


def intersect_lines(line1, line2, eps=1e-10):
    rho1, theta1 = line1
    rho2, theta2 = line2

    theta1_rad = np.deg2rad(theta1)
    theta2_rad = np.deg2rad(theta2)

    cos1, sin1 = np.cos(theta1_rad), np.sin(theta1_rad)
    cos2, sin2 = np.cos(theta2_rad), np.sin(theta2_rad)

    # Check for parallel lines using determinant
    det = cos1 * sin2 - sin1 * cos2
    if abs(det) < eps:
        # Lines are parallel
        return None

    x = (sin2 * rho1 - sin1 * rho2) / det
    y = (-cos2 * rho1 + cos1 * rho2) / det

    return x, y


def detect_chessboard_corners(gray_image, lines):
    im_h, im_w = gray_image.shape[:2]
    half_im_h, half_im_w = im_h // 2, im_w // 2
    corners_list = []

    num_lines = len(lines)
    for i in range(num_lines):
        for j in range(i + 1, num_lines):
            pt = intersect_lines(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                x += half_im_w
                y += half_im_h
                corners_list.append([x, y])

    if len(corners_list) == 0:
        return np.array([[], []])
    corners_array = np.array(corners_list).T
    return corners_array


def draw_lines(image, lines):
    img = np.copy(image)
    for line in lines:
        rho, theta, _, _ = line
        draw_line(img, rho, theta)
    return img


def draw_line(image, rho, theta):
    im_h, im_w = image.shape[:2]
    half_im_h, half_im_w = im_h // 2, im_w // 2
    line_x, line_y = sample_line_points(im_h, im_w, half_im_h, half_im_w, rho, theta)
    for i in range(line_x.size):
        x, y = int(line_x[i]), int(line_y[i])
        if 0 <= x < im_h and 0 <= y < im_w:
            image[x, y, :] = [0, 0, 255]


def draw_chessboard_lines(image, lines):
    img = np.copy(image)
    for line in lines:
        rho, theta = line
        draw_line(img, rho, theta)
    return img


def draw_corners(image, corners):
    img = np.copy(image)
    for i in range(corners.shape[1]):
        y, x = corners[:, i]
        cv2.circle(img, (int(round(y)), int(round(x))), 2, (0, 0, 255), -1)
    return img


def sample_line_points(im_h, im_w, half_im_h, half_im_w, rho, theta):
    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))
    if s == 0:
        x = np.arange(-im_h, im_h, 1) + half_im_h
        y = rho * np.ones((x.size)) + half_im_w
        return x, y
    y = np.arange(-im_w, im_w, 1)
    x = np.round(- c/s * y + rho/s) + half_im_h
    y = y + half_im_w
    return x, y


def hough_transform(image, edge_name, accumulator_name, lines_name, chess_lines_name, corners_name):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_h, im_w = image.shape[:2]
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    cv2.imwrite(edge_name, edges)
    chessboard_mask = np.zeros_like(edges)
    chess_x_s, chess_x_e, chess_y_s, chess_y_e = (im_h * 1) // 7, (im_h * 13) // 14, (im_w * 3) // 7, (im_w * 13) // 14
    chessboard_mask[chess_x_s:chess_x_e, chess_y_s:chess_y_e] = edges[chess_x_s:chess_x_e, chess_y_s:chess_y_e]
    accumulator, rho_values, theta_values = compute_hough_accumulator(chessboard_mask, num_rho=2000, num_theta=180)
    accumulator = cv2.normalize(accumulator, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(accumulator_name, accumulator)
    lines = extract_hough_lines(accumulator, rho_values, theta_values, threshold=60)
    cv2.imwrite(lines_name, draw_lines(image, lines))
    chess_lines = filter_chessboard_lines(gray_image, lines, chess_x_s, chess_x_e, chess_y_s, chess_y_e)
    cv2.imwrite(chess_lines_name, draw_chessboard_lines(image, chess_lines))
    corners = detect_chessboard_corners(gray_image, chess_lines)
    print(corners.shape)
    cv2.imwrite(corners_name, draw_corners(image, corners))


image_name = "im01"
image = cv2.imread("im01.jpg")
edge_image_name = "res01.jpg"
accumulator_image_name = "res03_hough_space.jpg"
lines_image_name = "res05_lines.jpg"
chessboard_lines_image_name = "res07_chess.jpg"
corners_image_name = "res09_corners.jpg"
hough_transform(image, edge_image_name, accumulator_image_name, lines_image_name, chessboard_lines_image_name, corners_image_name)


image_name = "im02"
image = cv2.imread("im02.jpg")
edge_image_name = "res02.jpg"
accumulator_image_name = "res04_hough_space.jpg"
lines_image_name = "res06_lines.jpg"
chessboard_lines_image_name = "res08_chess.jpg"
corners_image_name = "res10_corners.jpg"
hough_transform(image, edge_image_name, accumulator_image_name, lines_image_name, chessboard_lines_image_name, corners_image_name)

