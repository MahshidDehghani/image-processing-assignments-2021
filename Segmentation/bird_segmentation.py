import numpy as np
import skimage.segmentation
import cv2
from skimage.segmentation import mark_boundaries


def normalize_histogram(channel, mask, num_bins=256):
    histogram = np.zeros((num_bins))
    masked_pixel_indices = np.asarray(np.where(mask))
    num_masked_pixels = masked_pixel_indices.shape[1]
    for s in range(num_masked_pixels ):
        i, j = masked_pixel_indices[:, s]
        intensity = int(channel[i][j])
        histogram[intensity] += 1
    histogram[:] = histogram[:] / num_masked_pixels
    return histogram, num_masked_pixels


def histogram_distance(hist1, hist2):
    diff = hist1 - hist2
    squared_diff = np.power(diff, 2)
    distance = np.sum(squared_diff)
    return distance


def matches_any_histogram(query_histogram, reference_histograms, distance_threshold):
    for hist in reference_histograms:
        if histogram_distance(query_histogram, hist) < distance_threshold:
            return True
    return False


def extract_sample_bird_histograms(sample_bird_img):
    bird_segments = skimage.segmentation.felzenszwalb(sample_bird_img, scale=400, sigma=0.6, min_size=200)
    blue_channel = sample_bird_img[:, :, 0]
    green_channel = sample_bird_img[:, :, 1]
    red_channel = sample_bird_img[:, :, 2]
    mask = bird_segments > 0
    blue_hist, _ = normalize_histogram(blue_channel, mask)
    green_hist, _ = normalize_histogram(green_channel, mask)
    red_hist, _ = normalize_histogram(red_channel, mask)
    return bird_segments, blue_hist, green_hist, red_hist


def detect_birds(main_image, sample_bird_images, histogram_distance_threshold):
    sample_segments, sample_histograms = [], []
    for bird in sample_bird_images:
        sample_segment, blue_hist, green_hist, red_hist = extract_sample_bird_histograms(bird)
        sample_segments.append(sample_segment)
        sample_histograms.append(blue_hist + green_hist + red_hist)
    main_blue_channel = main_image[:, :, 0]
    main_green_channel = main_image[:, :, 1]
    main_red_channel = main_image[:, :, 2]
    bird_labels = []
    main_segments = skimage.segmentation.felzenszwalb(main_image, scale=600, sigma=1.3, min_size=1500)
    cv2.imwrite("segmentation_boundaries.jpg", cv2.normalize(mark_boundaries(main_image, main_segments), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    num_labels = np.max(main_segments)
    for label in range(num_labels):
        segment_mask = main_segments == label
        segment_size = np.asarray(np.where(segment_mask)).shape[1]
        if segment_size < 100000:
            blue_hist, _ = normalize_histogram(main_blue_channel, segment_mask)
            green_hist, _ = normalize_histogram(main_green_channel, segment_mask)
            red_hist, _ = normalize_histogram(main_red_channel, segment_mask)
            if matches_any_histogram(blue_hist + green_hist + red_hist, sample_histograms, histogram_distance_threshold):
                bird_labels.append(label)
    result = np.zeros_like(main_image)
    for label in bird_labels:
        segment_pixels = main_segments == label
        result[segment_pixels ] = main_image[segment_pixels ]
    return result


histogram_distance_threshold = 0.1
main_image = cv2.imread("birds.jpg")
im_h, im_w = main_image.shape[:2]
cropped_image = np.zeros_like(main_image)
cropped_image[(3 * im_h) // 7:(5 * im_h) // 7, :, :] = main_image[(3 * im_h) // 7:(5 * im_h) // 7, :, :]
sample_bird1_img = cv2.imread("bird1.jpg")
sample_bird2_img = cv2.imread("bird2.jpg")
detected_birds_img = detect_birds(cropped_image, [sample_bird1_img, sample_bird2_img], histogram_distance_threshold)
cv2.imwrite("res10.jpg", detected_birds_img)
