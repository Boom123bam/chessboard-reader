from findchessboard import find_chessboard
from flask import Flask, request
import numpy as np
import cv2

app = Flask(__name__)


@app.route("/getchessboard", methods=["GET", "POST"])
def get_chessboard():
    if request.method == "POST":
        # Get the image file
        image_file = request.files["file"]
        # Read the image file using cv2
        image_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        [found, a, b] = find_chessboard(image)
        if found:
            return str(a) + str(b)
        else:
            return "No board"
    else:
        return """
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
        </form>
        """


if __name__ == "__main__":
    app.run(debug=True)


# if __name__ == "__main__":
#     # read grayscale
#     # original = cv2.imread("test-images/pic.jpg", cv2.IMREAD_GRAYSCALE)
#     # original = cv2.imread("test-images/mirror.jpg", cv2.IMREAD_GRAYSCALE)
#     # original = cv2.imread("test-images/board.png", cv2.IMREAD_GRAYSCALE)
#     # original = cv2.imread("test-images/respawn.png", cv2.IMREAD_GRAYSCALE)
#     original = cv2.imread("test-images/respawn2.png", cv2.IMREAD_GRAYSCALE)
#     # original = cv2.imread("test-images/board-light.png", cv2.IMREAD_GRAYSCALE)
#     # original = cv2.imread("test-images/example.jpeg", cv2.IMREAD_GRAYSCALE)

#     h, w = original.shape
#     board_x_lines, board_y_lines = find_chessboard(original)

#     plt.imshow(original, cmap="gray")
#     for hx in board_x_lines:
#         plt.axhline(round(hx * h), color="b", lw=2)

#     for hy in board_y_lines:
#         plt.axvline(round(hy * w), color="r", lw=2)

#     plt.waitforbuttonpress()
