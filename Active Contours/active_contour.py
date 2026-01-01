import sys
import cv2
import numpy as np
from moviepy.editor import *


def on_mouse_click(event, x, y, flags, params):
    global initial_points_image, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([int(y), int(x)])
        cv2.circle(initial_points_image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('image', initial_points_image)


def select_initial_contour(image):
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', on_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def initialize_circular_snake(center_x, center_y, radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.zeros((num_points, 2), dtype='int32')
    points[:, 0] = np.array(center_x + radius * np.sin(angles))
    points[:, 1] = np.array(center_y + radius * np.cos(angles))
    return points


def gradient_magnitude(gray_image):
    return cv2.Canny(gray_image, 100, 200)


def contour_length_squared(num_points, points):
    d = np.zeros((num_points, 2))
    d[0:num_points - 1, :] = points[0:num_points - 1, :] - points[1:num_points, :]     # [x_i - x_i+1, y_i - y_i+1]
    d[num_points - 1, :] = points[num_points - 1, :] - points[0, :]             # [x_n-1 - x_0, y_n-1 - y_0]
    d = np.power(d, 2)          # [(x_i - x_i+1)^2, (y_i - y_i+1)^2]
    d = d[:, 0] + d[:, 1]       # [(x_i - x_i+1)^2 + (y_i - y_i+1)^2]
    return np.sum(np.sqrt(d))


def snake_normal_i(v_i_mines_one, v_i_plus_one):
    x_i_mines_one, y_i_mines_one = v_i_mines_one
    x_i_plus_one, y_i_plus_one = v_i_plus_one
    dx = x_i_plus_one - x_i_mines_one
    dy = y_i_plus_one - y_i_mines_one
    if dx == 0 and dy == 0:
        return [0, 0]
    if dx == 0:
        return [1, 0]
    if dy == 0:
        return [0, -1]
    min_abs_diff = min(abs(dx), abs(dy))
    dx, dy = int(dx/min_abs_diff), int(dy/min_abs_diff)
    return [(dy), -(dx)]


def snake_normals(num_points, points):
    normals = np.zeros_like(points)
    for i in range(num_points):
        v_i_plus_one = points[(i + 1) % num_points, :]
        v_i_mines_one = points[(i - 1) % num_points, :]
        normals[i, :] = snake_normal_i(v_i_mines_one, v_i_plus_one)
    return normals


def external_energy_i(image_gradient, v_i):
    x_i, y_i = v_i
    return -1 * image_gradient[x_i, y_i]


def elastic_energy_i(d, v_i, v_i_plus_one):
    x_i, y_i = v_i
    x_i_plus_one, y_i_plus_one = v_i_plus_one
    e = ((x_i_plus_one - x_i) ** 2 + (y_i_plus_one - y_i) ** 2 - d) ** 2
    return e


def smooth_energy_i(v_i_mines_one, v_i, v_i_plus_one):
    x_i_mines_one, y_i_mines_one = v_i_mines_one
    x_i, y_i = v_i
    x_i_plus_one, y_i_plus_one = v_i_plus_one
    e = (x_i_plus_one - 2 * x_i + x_i_mines_one) ** 2 + (y_i_plus_one - 2 * y_i + y_i_mines_one) ** 2
    return e


def internal_energy_i(d, alpha, beta, v_i_mines_one, v_i, v_i_plus_one):
    return alpha * elastic_energy_i(d, v_i, v_i_plus_one) + beta * smooth_energy_i(v_i_mines_one, v_i, v_i_plus_one)


def energy_i(image_gradient, d, alpha, beta, gamma, v_i_mines_one, v_i, v_i_plus_one):
    internal_e = internal_energy_i(d, alpha, beta, v_i_mines_one, v_i, v_i_plus_one)
    eternal_e = external_energy_i(image_gradient, v_i)
    return internal_e + gamma * eternal_e


def contour_displacement(points, prev_points):
    d = points - prev_points
    d = np.power(d, 2)
    d = d[:, 0] + d[:, 1]
    d = np.sqrt(d)
    return np.sum(d)


def greedy_snake_step(image_gradient, num_points, points, locked_points, window_size, d, alpha, beta, gamma):
    r = window_size // 2
    updated_points = np.copy(points)
    energy = 0
    normals = snake_normals(num_points, points)
    for i in range(0, num_points):
        min_energy = sys.maxsize
        v_i = points[i, :]
        v_i_plus_one = points[(i + 1) % num_points, :]
        v_i_mines_one = points[(i - 1) % num_points, :]
        dx_i, dy_i = normals[i, :]
        if not locked_points[i]:
            for j in range(window_size):
                candidate_v_i = [v_i[0] + (j - r) * dx_i, v_i[1] + (j - r) * dy_i]
                internal_e = internal_energy_i(d, alpha, beta, v_i_mines_one, candidate_v_i, v_i_plus_one)
                external_e = external_energy_i(image_gradient, candidate_v_i)
                e_i = internal_e + gamma * external_e
                if e_i < min_energy:
                    e_e = external_e
                    min_energy = e_i
                    updated_points[i, :] = candidate_v_i
            if e_e < 0:
                locked_points[i] = True
        else:
            internal_e = internal_energy_i(d, alpha, beta, v_i_mines_one, v_i, v_i_plus_one)
            external_e = external_energy_i(image_gradient, v_i)
            min_energy = internal_e + gamma * external_e
        energy += min_energy
    return updated_points, energy


def dp_snake_step(image_gradient, num_points, points, locked_points, window_size, d, alpha, beta, gamma):
    r = window_size // 2
    updated_points = np.zeros_like(points)
    energy_table = np.zeros((num_points, window_size, 2))
    normals = snake_normals(num_points, points)
    for i in range(0, num_points):
        v_i = points[i, :]
        v_i_plus_one = points[(i + 1) % num_points, :]
        v_i_mines_one = points[(i - 1) % num_points, :]
        dx_i, dy_i = normals[i, :]
        dx_i_plus_one, dy_i_plus_one = normals[(i + 1) % num_points, :]
        if locked_points[i] and locked_points[(i + 1) % num_points]:
            window_idx_i, window_idx_i_plus_one = r, r
            e_i = energy_i(image_gradient, d, alpha, beta, gamma, v_i_mines_one, v_i, v_i_plus_one) + energy_table[(i + 1) % num_points, window_idx_i_plus_one, 0]
            min_energy = e_i
            energy_table[i, window_idx_i, :] = [e_i, window_idx_i_plus_one]
        elif locked_points[i]:
            min_energy = sys.maxsize
            window_idx_i = r
            for window_idx_i_plus_one in range(window_size):
                candidate_v_i_plus_one = [v_i_plus_one[0] + (window_idx_i_plus_one - r) * dx_i_plus_one,
                                    v_i_plus_one[1] + (window_idx_i_plus_one - r) * dy_i_plus_one]
                internal_e = internal_energy_i(d, alpha, beta, v_i_mines_one, v_i, candidate_v_i_plus_one)
                external_e = external_energy_i(image_gradient, v_i)
                e_i = internal_e + gamma * external_e + energy_table[(i + 1) % num_points, window_idx_i_plus_one, 0]
                if e_i < min_energy:
                    energy_table[i, window_idx_i, :] = [e_i, window_idx_i_plus_one]
                    min_energy = e_i
        else:
            for window_idx_i in range(window_size):
                min_energy = sys.maxsize
                candidate_v_i = [v_i[0] + (window_idx_i - r) * dx_i, v_i[1] + (window_idx_i - r) * dy_i]
                for window_idx_i_plus_one in range(window_size):
                    candidate_v_i_plus_one = [v_i_plus_one[0] + (window_idx_i_plus_one - r) * dx_i_plus_one,
                                        v_i_plus_one[1] + (window_idx_i_plus_one - r) * dy_i_plus_one]
                    internal_e = internal_energy_i(d, alpha, beta, v_i_mines_one, candidate_v_i, candidate_v_i_plus_one)
                    external_e = external_energy_i(image_gradient, candidate_v_i)
                    e_i = internal_e + gamma * external_e + energy_table[(i + 1) % num_points, window_idx_i_plus_one, 0]
                    if e_i < min_energy:
                        e_e = external_e
                        energy_table[i, window_idx_i, :] = [e_i, window_idx_i_plus_one]
                        min_energy = e_i
                if e_e < 0:
                    locked_points[i] = True

    j = int(np.argmin(energy_table[num_points - 1, :, 0]))
    dx_n, dy_n = normals[num_points - 1, :]
    updated_points[num_points - 1, :] = [points[num_points - 1, 0] + (j - r) * dx_n, points[num_points - 1, 1] + (j - r) * dy_n]
    if locked_points[num_points - 1]:
        updated_points[num_points - 1, :] = points[num_points - 1, :]
        j = r
    j = int(energy_table[num_points - 1, j, 1])
    for i in range(num_points - 1):
        dx_i, dy_i = normals[i, :]
        updated_points[i, :] = [points[i, 0] + (j - r) * dx_i, points[i, 1] + (j - r) * dy_i]
        if locked_points[i]:
            updated_points[i, :] = points[i, :]
            j = r
        j = int(energy_table[i, j, 1])
    return updated_points, min_energy


def active_contour(image, initial_points, window_size, elasticity_scale, alpha, beta, gamma, max_iteration, convergence_threshold):
    frames = []

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gradient = gradient_magnitude(gray_image).astype(np.float32)

    num_points = initial_points.shape[0]
    locked_points = np.zeros(num_points, dtype=bool)

    current_points = np.copy(initial_points)
    prev_energy = sys.maxsize
    for _ in range(max_iteration):
        result = draw_snake(image, current_points)
        frames.append(result)
        squared_length = contour_length_squared(num_points, initial_points)
        d = (squared_length / num_points) * elasticity_scale
        updated_points, total_energy = greedy_snake_step(image_gradient, num_points, current_points, locked_points, window_size, d, alpha, beta, gamma)
        if contour_displacement(updated_points, current_points) < convergence_threshold:
            break
        prev_energy = total_energy
        current_points = np.copy(updated_points)
    return result, frames


def draw_snake(orginal_image, points):
    img = np.copy(orginal_image)
    prev_x, prev_y = None, None
    for i in range(points.shape[0]):
        x, y = points[i, :]
        cv2.circle(img, (y, x), 3, (0, 255, 0), -1)
        if prev_x is not None:
            cv2.line(img, (prev_y, prev_x), (y, x), [0, 255, 0], 1)
        prev_x, prev_y = x, y
    x, y = points[0, :]
    cv2.line(img, (prev_y, prev_x), (y, x), [0, 255, 0], 1)
    return img


def create_video(output_path, initial_contour_image, frames, fps=10):
    video_clips = []
    initial_contour_path = "videoFrames/points_image.jpg"
    cv2.imwrite(initial_contour_path, initial_contour_image)
    video_clips.append(ImageClip(initial_contour_path).set_duration(1))
    frame_idx = 0
    for frame in frames:
        initial_contour_path = "videoFrames/frame" + str(frame_idx) + ".jpg"
        cv2.imwrite(initial_contour_path, frame)
        video_clips.append(ImageClip(initial_contour_path).set_duration(0.2))
        frame_idx += 1
    final_video = concatenate_videoclips(video_clips, method='compose')
    final_video.write_videofile(output_path, fps=fps, remove_temp=True, codec="libx264", audio_codec="aac")


image = cv2.imread("tasbih.jpg")

search_window_size = 5
elasticity_scale = 0.001
alpha = 0.007
beta = 0.0000003
gamma = 10
num_iterations = 100
convergence_threshold = 4

clicked_points = []
initial_points_image = np.copy(image)
select_initial_contour(initial_points_image)
circle_center_x, circle_center_y = clicked_points[0]
circle_edge_x, circle_edge_y = clicked_points[1]
radius = np.sqrt((circle_edge_x - circle_center_x) ** 2 + (circle_edge_y - circle_center_y) ** 2)
cv2.imwrite("points.jpg", initial_points_image)
initial_snake_points = initialize_circular_snake(circle_center_x, circle_center_y, radius, 60)

result, frames = active_contour(image, initial_snake_points, search_window_size, elasticity_scale, alpha, beta, gamma, num_iterations, convergence_threshold)
cv2.imwrite("res11.jpg", result)
output_video_path = "contour.mp4"
create_video(output_video_path, initial_points_image, frames)