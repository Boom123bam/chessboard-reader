from findchessboard import find_chessboard
from matplotlib import pyplot as plt
import cv2


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
