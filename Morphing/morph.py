from PIL import Image
import cv2
import numpy as np
import scipy.spatial as scp


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


def save_points_to_file(points, file_path):
    file = open(file_path, "w")
    num_points = len(points)
    num_points_str = str(num_points)
    file.write(num_points_str+'\n')
    for point in points:
        x, y = point
        file.write(str(x)+' '+str(y)+'\n')
    file.close()


def load_points_from_file(file_path):
    file = open(file_path, 'r')
    num_points_str = file.readline()
    num_points = int(num_points_str.strip())
    points = np.zeros((num_points, 2))
    i = 0
    while True:
        line = file.readline()
        if not line:
            break
        x_str, y_str = line.split(" ")
        x, y = int(x_str.strip()), int(y_str.strip())
        points[i, :] = [x, y]
        i += 1
    return num_points, points


def compute_delaunay_triangles(points):
    triangles = scp.Delaunay(points)
    return triangles.simplices


def draw_triangles(image, tri_vertices, points):
    tri_image = np.copy(image)
    for i in range(tri_vertices.shape[0]):
        v1 = points[tri_vertices[i, 0], :]
        v2 = points[tri_vertices[i, 1], :]
        v3 = points[tri_vertices[i, 2], :]
        cv2.line(tri_image, (v1[0], v1[1]), (v2[0], v2[1]), (0, 20, 255), 1)
        cv2.line(tri_image, (v1[0], v1[1]), (v3[0], v3[1]), (0, 20, 255), 1)
        cv2.line(tri_image, (v2[0], v2[1]), (v3[0], v3[1]), (0, 20, 255), 1)
    return tri_image


def create_mask(h, w, dst_pts):
    mask = np.zeros((h, w, 3), dtype='uint8')
    cv2.fillPoly(mask, pts=[dst_pts.astype('int')], color=(1, 1, 1))
    return mask


def warp_triangle(h, w, src_image, dst_vertices, src_points, v1_idx, v2_idx, v3_idx):
    src_vertices = np.float32([src_points[v1_idx, :], src_points[v2_idx, :], src_points[v3_idx, :]])
    affine_transform = cv2.getAffineTransform(src_vertices, dst_vertices)
    warped = cv2.warpAffine(src_image, affine_transform, (w, h)).astype('float32')
    return warped


def warp(im1, im2, tri_vertices, points1, points2, interpolated_points):
    h, w = im1.shape[:2]
    im1_warped = np.zeros((h, w, 3))
    im2_warped = np.zeros((h, w, 3))
    for i in range(tri_vertices.shape[0]):
        v1_index = tri_vertices[i, 0]
        v2_index = tri_vertices[i, 1]
        v3_index = tri_vertices[i, 2]
        dst_vertices = np.float32(
            [interpolated_points[v1_index, :], interpolated_points[v2_index, :], interpolated_points[v3_index, :]])
        mask = create_mask(h, w, dst_vertices)
        im1_mask = np.equal(im1_warped, 0)
        im2_mask = np.equal(im1_warped, 0)
        im1_warped += warp_triangle(h, w, im1, dst_vertices, points1, v1_index, v2_index, v3_index) * mask * im1_mask
        im2_warped += warp_triangle(h, w, im2, dst_vertices, points2, v1_index, v2_index, v3_index) * mask * im2_mask
        # im1_warped += warp_triangle(h, w, im1, dst_vertices, points1, v1_index, v2_index, v3_index) * mask
        # im2_warped += warp_triangle(h, w, im2, dst_vertices, points2, v1_index, v2_index, v3_index) * mask
    return im1_warped, im2_warped


def morph(im1, im2, points1, points2, num_frames):
    num_frames -= 1
    frames = []
    tri_vertices = compute_delaunay_triangles(points1)
    for i in range(num_frames + 1):
        alpha = i / num_frames
        interpolated_points = (alpha * points2 + (1 - alpha) * points1).astype('int')
        im1_warped, im2_warped = warp(im1, im2, tri_vertices, points1, points2, interpolated_points)
        morphed_frame= (alpha * im2_warped + (1 - alpha) * im1_warped).astype('uint8')
        if i == 14:
            frame_15 = morphed_frame
        if i == 29:
            frame_30 = morphed_frame
        frames.append(morphed_frame)
    return frames, frame_15, frame_30


def create_gif(gif_path, frames, fps=30):
    frame_names = []
    i = 0
    for frame in frames:
        name = "videoFrames/frame" + str(i) + ".jpg"
        cv2.imwrite(name, frame)
        frame_names.append(name)
        i += 1
    reverse_i = i
    while i > 0:
        i -= 1
        name = "videoFrames/frame" + str(reverse_i) + ".jpg"
        cv2.imwrite(name, cv2.imread("videoFrames/frame" + str(i) + ".jpg"))
        frame_names.append(name)
        reverse_i += 1
    d = 1000 / fps
    frames = [Image.open(image) for image in frame_names]
    frame_one = frames[0]
    frame_one.save(gif_path, format="GIF", append_images=frames,
                   save_all=True, duration=d, loop=0)


im1 = cv2.imread("res01.jpg")
im2 = cv2.imread("res02.jpg")
h, w = im1.shape[:2]
im2 = cv2.resize(im2, (w, h))

# clicked_points = []
# points_image = np.copy(im1)
# select_points(im1)
# # write_points(clicked_points, "im1_points.txt")
#
# clicked_points = []
# points_image = np.copy(im2)
# select_points(im2)
# # write_points(clicked_points, "im2_points.txt")

_, points1 = load_points_from_file("im1_points.txt")
_, points2 = load_points_from_file("im2_points.txt")

morph_frames, frame_15, frame_30 = morph(im1, im2, points1, points2, num_frames=45)
cv2.imwrite("res03.jpg", frame_15)
cv2.imwrite("res04.jpg", frame_30)
create_gif("morph.gif", morph_frames)


