import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve


def on_mouse_click(event, x, y, flags, params):
    global points_image, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([int(x), int(y)])
        cv2.circle(points_image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('image', points_image)


def select_points(image):
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', on_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_contour(image, points):
    img = np.copy(image)
    prev_x, prev_y = None, None
    for i in range(len(points)):
        x, y = points[i]
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        if prev_x is not None:
            cv2.line(img, (prev_x, prev_y), (x, y), [0, 255, 0], 1)
        prev_x, prev_y = x, y
    x, y = points[0]
    cv2.line(img, (prev_x, prev_y), (x, y), [0, 255, 0], 1)
    return img


def place_resized_source_on_target(target_image, source_image, source_size, top_left_position):
    source = np.zeros_like(target_image)
    x, y = top_left_position
    w, h = source_size
    source[x:x + h, y:y + w, :] = cv2.resize(source_image, source_size)
    return source


def generate_mask_from_points(target_image, points):
    mask_image = np.zeros_like(target_image)
    cv2.fillPoly(mask_image, pts=[points], color=(255, 255, 255))
    mask = np.zeros((target_image.shape[0], target_image.shape[1]), dtype=bool)
    mask[mask_image[:, :, 0] > 0] = True
    return mask_image, mask


def get_mask_bounding_box(mask, padding=1):
    x, y = np.where(mask)
    x_start, x_end = np.min(x), np.max(x) + 1
    y_start, y_end = np.min(y), np.max(y) + 1
    return x_start - padding, x_end + padding, y_start - padding, y_end + padding


def poisson_matrix(mask):
    h, w = mask.shape
    n = h * w
    A = scipy.sparse.identity(n, format='lil')
    x, y = np.where(mask)
    idx = y + x * w
    for i in idx:
        A[i, i] = 4
        if i - 1 >= 0:
            A[i, i - 1] = -1
        if i + 1 < n:
            A[i, i + 1] = -1
        if i - w >= 0:
            A[i, i - w] = -1
        if i + h < n:
            A[i, i + w] = -1
    return A.tocsr()


def poisson_rhs(target_channel, source_channel, mask):
    h, w = mask.shape
    target_vector = target_channel.flatten()
    b = cv2.filter2D(source_channel, cv2.CV_64F, laplacian_filter)
    b = b.flatten()
    x, y = np.where(mask == False)
    idx = y + w * x
    b[idx] = target_vector[idx]
    return b


def solve_equation(A, b, mask):
    h, w = mask.shape
    x = spsolve(A, b)
    x = np.reshape(x, (h, w))
    x[x > 255] = 255
    x[x < 0] = 0
    x = x.astype(int)
    return x


def poisson_blend(source_image, target_image, blend_mask):
    output_image = np.copy(target_image)
    x_start, x_end, y_start, y_end = get_mask_bounding_box(blend_mask)
    cropped_mask = blend_mask[x_start:x_end, y_start:y_end]
    A = poisson_matrix(cropped_mask)
    for color in range(target_image.shape[2]):
        target_channel = target_image[x_start:x_end, y_start:y_end, color]
        source_channel = source_image[x_start:x_end, y_start:y_end, color]
        b = poisson_rhs(target_channel, source_channel, cropped_mask)
        x = solve_equation(A, b, cropped_mask)
        output_image[x_start:x_end, y_start:y_end, color] = x
    return output_image


laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

background_image = cv2.imread("res06.jpg")
foreground_image = cv2.imread("res05.jpg")
foreground_size = (305, 221)
foreground_position = (375, 100)
source_image = place_resized_source_on_target(background_image, foreground_image, foreground_size, foreground_position)
cv2.imwrite("target.jpg", source_image)

clicked_points = [[149, 591], [159, 569], [157, 546], [159, 518], [160, 498], [179, 478], [191, 454], [205, 428], [229, 411], [262, 402], [296, 405], [313, 424], [322, 443], [329, 470], [329, 488], [355, 492], [383, 500], [399, 512], [401, 594]]
# clicked_points = []
# points_image = np.copy(source)
# select_points(source)
contour_image = draw_contour(np.copy(source_image), clicked_points)
cv2.imwrite("contour.jpg", contour_image)

mask_img, mask = generate_mask_from_points(source_image, np.asarray(clicked_points))
cv2.imwrite("mask.jpg", mask_img)

background_image[mask] = source_image[mask]
cv2.imwrite("without_poisson_blending.jpg", background_image)

blended_image = poisson_blend(source_image, background_image, mask)
cv2.imwrite("res07.jpg", blended_image)
