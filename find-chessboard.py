import numpy as np
import cv2
from matplotlib import pyplot as plt
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
    rows, cols = img.shape
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
        gapsdiffs, lambda x: abs(x) < 3
    )  # allow 3px error in both directions
    return peaks[start : end + 2]


def find_chessboard(img):
    # rescale and blur
    width = 750
    height = 750
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    img = opening(resized, 3)
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

    # Color for showing lines
    cdstX = cv2.cvtColor(sobXBinary, cv2.COLOR_GRAY2BGR)
    cdstY = cv2.cvtColor(sobYBinary, cv2.COLOR_GRAY2BGR)

    if linesHorizontal is not None:
        for i in range(0, len(linesHorizontal)):
            rho = linesHorizontal[i][0][0]
            theta = linesHorizontal[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv2.line(cdstX, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    if linesVertical is not None:
        for i in range(0, len(linesVertical)):
            rho = linesVertical[i][0][0]
            theta = linesVertical[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv2.line(cdstY, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

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
        resized = rotate(resized, avg_rotation * 180 / math.pi)

    # Separate gradients to + and -

    dx_pos = np.clip(sobX, 0, 255)
    # dx_pos = dilate(dx_pos, 1)
    dx_neg = np.clip(sobX, -255, 0)
    dy_pos = np.clip(sobY, 0, 255)
    # dy_pos = dilate(dy_pos, 1)
    dy_neg = np.clip(sobY, -255, 0)

    dx_pos_sum = np.sum(dx_pos, axis=1)
    dx_pos_sum = np.clip(dx_pos_sum, 0, np.max(dx_pos_sum) / 3)
    dx_neg_sum = np.sum(-dx_neg, axis=1)
    dx_neg_sum = np.clip(dx_neg_sum, 0, np.max(dx_neg_sum) / 3)

    dy_pos_sum = np.sum(dy_pos, axis=0)
    dy_pos_sum = np.clip(dy_pos_sum, 0, np.max(dy_pos_sum) / 3)
    dy_neg_sum = np.sum(-dy_neg, axis=0)
    dy_neg_sum = np.clip(dy_neg_sum, 0, np.max(dy_neg_sum) / 3)

    hough_dx = dx_pos_sum * dx_neg_sum
    hough_dy = dy_pos_sum * dy_neg_sum

    # Find peaks

    dx_peaks, properties = find_peaks(
        hough_dx, distance=30, height=np.max(hough_dx) / 10
    )
    dy_peaks, _ = find_peaks(hough_dy, distance=30, height=np.max(hough_dy) / 10)

    dx_peaks = prune_non_equally_spaced(dx_peaks)
    dy_peaks = prune_non_equally_spaced(dy_peaks)

    assert len(dx_peaks) == 7 and len(dy_peaks) == 7, "chessboard not found"

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
    board_x_lines /= 750
    board_y_lines = board_y_lines.astype(np.float64)
    board_y_lines /= 750

    return board_x_lines, board_y_lines


if __name__ == "__main__":
    # read grayscale
    # original = cv2.imread("test-images/pic.jpg", cv2.IMREAD_GRAYSCALE)
    # original = cv2.imread("test-images/mirror.jpg", cv2.IMREAD_GRAYSCALE)
    # original = cv2.imread("test-images/board.png", cv2.IMREAD_GRAYSCALE)
    # original = cv2.imread("test-images/respawn.png", cv2.IMREAD_GRAYSCALE)
    original = cv2.imread("test-images/respawn2.png", cv2.IMREAD_GRAYSCALE)
    # original = cv2.imread("test-images/board-light.png", cv2.IMREAD_GRAYSCALE)
    # original = cv2.imread("test-images/example.jpeg", cv2.IMREAD_GRAYSCALE)

    h, w = original.shape
    board_x_lines, board_y_lines = find_chessboard(original)

    plt.imshow(original, cmap="gray")
    for hx in board_x_lines:
        plt.axhline(round(hx * h), color="b", lw=2)

    for hy in board_y_lines:
        plt.axvline(round(hy * w), color="r", lw=2)

    plt.waitforbuttonpress()
