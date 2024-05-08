import numpy as np
import cv2
import math
from scipy.signal import find_peaks


def erode(img, size):
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1), (size, size)
    )
    return cv2.erode(img, element)


def dilate(img, size):
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1), (size, size)
    )
    return cv2.dilate(img, element)


def opening(x, size=3):
    return dilate(erode(x, size), size)


def closing(x, size=3):
    return erode(dilate(x, size), size)


def sobelY(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)


def sobelX(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)


def toAbs(img):
    abs_img = np.absolute(img)
    return np.uint8(abs_img)


def rotate(img, angle):
    rows, cols = img.shape[0], img.shape[1]
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def find_longest_chain(list, condition):
    """find longest chain of elements in list that satisfy condition and return start and end index"""
    max_len = 0
    current_len = 0
    max_index = 0
    for i in range(len(list)):
        if condition(list[i]):
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_index = i - current_len
            current_len = 0

    if current_len > max_len:
        max_len = current_len
        max_index = i - current_len + 1

    return max_index, max_index + max_len


def prune_non_equally_spaced(peaks):
    gaps = np.diff(peaks)
    gapsdiffs = np.diff(gaps)
    # find longest seqence of small diffs
    start, end = find_longest_chain(
        gapsdiffs, lambda x: abs(x) < 5
    )  # allow 3px error in both directions
    return peaks[start : end + 2]


def find_chessboard(original_img, saveTiles):
    # grayscale
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # rescale and blur
    width = 750
    height = 750
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    img = opening(img, 3)
    img = closing(img, 5)
    # img = cv2.medianBlur(img, 5)
    sobY = sobelY(img)
    sobX = sobelX(img)

    # Convert to binary
    _, sobXBinary = cv2.threshold(
        toAbs(sobX), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, sobYBinary = cv2.threshold(
        toAbs(sobY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    #  Standard Hough Line Transform
    linesHorizontal = cv2.HoughLines(sobXBinary, 1, np.pi / 180, 150, None, 0, 0)
    linesVertical = cv2.HoughLines(sobYBinary, 1, np.pi / 180, 150, None, 0, 0)

    # Find the average rotation
    angle_threshold = 0.1
    sum = 0
    count = 0
    if linesHorizontal is not None:
        for line in linesHorizontal:
            angle = line[0][1]
            angle -= math.pi / 2
            # filter angles within threshold
            if angle > angle_threshold / 2 or angle < -angle_threshold / 2:
                continue
            sum += angle
            count += 1

    if linesVertical is not None:
        for line in linesVertical:
            angle = line[0][1]
            if angle > math.pi / 2:
                angle -= math.pi
            # filter angles within threshold
            if angle > angle_threshold / 2 or angle < -angle_threshold / 2:
                continue
            sum += angle
            count += 1

    avg_rotation = sum / count

    # Rotate sobel X and Y

    if abs(avg_rotation) > 0.01:
        sobX = rotate(sobX, avg_rotation * 180 / math.pi)
        sobY = rotate(sobY, avg_rotation * 180 / math.pi)
        original_img = rotate(original_img, avg_rotation * 180 / math.pi)

    # Separate gradients to + and -

    dx_pos = np.clip(sobX, 0, 255)
    # dx_pos = dilate(dx_pos, 1)
    dx_neg = np.clip(sobX, -255, 0)
    dy_pos = np.clip(sobY, 0, 255)
    # dy_pos = dilate(dy_pos, 1)
    dy_neg = np.clip(sobY, -255, 0)

    dx_pos_sum = np.sum(dx_pos, axis=1)
    dx_neg_sum = np.sum(-dx_neg, axis=1)

    dy_pos_sum = np.sum(dy_pos, axis=0)
    dy_neg_sum = np.sum(-dy_neg, axis=0)

    dx_pos_sum = np.clip(dx_pos_sum, 0, np.max(dx_pos_sum) / 2)
    dx_neg_sum = np.clip(dx_neg_sum, 0, np.max(dx_neg_sum) / 2)
    dy_pos_sum = np.clip(dy_pos_sum, 0, np.max(dy_pos_sum) / 2)
    dy_neg_sum = np.clip(dy_neg_sum, 0, np.max(dy_neg_sum) / 2)

    dx_pos_sum[dx_pos_sum < np.max(dx_pos_sum) / 10] = 0
    dx_neg_sum[dx_neg_sum < np.max(dx_neg_sum) / 10] = 0
    dy_pos_sum[dy_pos_sum < np.max(dy_pos_sum) / 10] = 0
    dy_neg_sum[dy_neg_sum < np.max(dy_neg_sum) / 10] = 0

    hough_dx = dx_pos_sum * dx_neg_sum
    hough_dy = dy_pos_sum * dy_neg_sum

    # Find peaks

    dx_peaks, _ = find_peaks(hough_dx, distance=30, height=np.max(hough_dx) / 8)
    dy_peaks, _ = find_peaks(hough_dy, distance=30, height=np.max(hough_dy) / 8)

    dx_peaks = prune_non_equally_spaced(dx_peaks)
    dy_peaks = prune_non_equally_spaced(dy_peaks)

    # if more than 7 peaks, prune the smallest if it is the first or last peak
    lowest_dy_peak_index = np.argmin(hough_dy[dy_peaks])
    while len(dy_peaks) > 7 and (
        lowest_dy_peak_index == 0 or lowest_dy_peak_index == len(dy_peaks) - 1
    ):
        dy_peaks = np.delete(dy_peaks, lowest_dy_peak_index)
        lowest_dy_peak_index = np.argmin(hough_dy[dy_peaks])

    lowest_dx_peak_index = np.argmin(hough_dx[dx_peaks])
    while len(dx_peaks) > 7 and (
        lowest_dx_peak_index == 0 or lowest_dx_peak_index == len(dx_peaks) - 1
    ):
        dx_peaks = np.delete(dx_peaks, lowest_dx_peak_index)
        lowest_dx_peak_index = np.argmin(hough_dx[dx_peaks])

    # assert len(dx_peaks) == 7 and len(dy_peaks) == 7, "chessboard not found"
    if len(dx_peaks) != 7 or len(dy_peaks) != 7:
        return None

    # Estimate first and last lines

    avg_gap_x = round(np.average(np.diff(dx_peaks)))
    avg_gap_y = round(np.average(np.diff(dy_peaks)))

    board_x_lines = np.concatenate(
        (
            [max(dx_peaks[0] - avg_gap_x, 0)],
            dx_peaks,
            [min(dx_peaks[-1] + avg_gap_x, height)],
        )
    )
    board_y_lines = np.concatenate(
        (
            [max(dy_peaks[0] - avg_gap_y, 0)],
            dy_peaks,
            [min(dy_peaks[-1] + avg_gap_y, width)],
        )
    )

    # normalize
    board_x_lines = board_x_lines.astype(np.float64)
    board_x_lines /= width
    board_y_lines = board_y_lines.astype(np.float64)
    board_y_lines /= height

    w, h, _ = original_img.shape
    x1 = round(board_x_lines[0] * w)
    x2 = round(board_x_lines[-1] * w)
    y1 = round(board_y_lines[0] * h)
    y2 = round(board_y_lines[-1] * h)
    result = original_img[x1:x2, y1:y2]

    if saveTiles:
        i = 0
        for x in range(8):
            for y in range(8):
                x1 = round(board_x_lines[x] * w)
                x2 = round(board_x_lines[x + 1] * w)
                y1 = round(board_y_lines[y] * h)
                y2 = round(board_y_lines[y + 1] * h)
                cv2.imwrite(
                    f"temp-board/{'{:02d}'.format(i)}.png", original_img[x1:x2, y1:y2]
                )
                # tiles.append(boardImg[x1:x2, y1:y2])
                i += 1

    return result
