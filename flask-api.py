from findchessboard import find_chessboard
from processboard import splitBoard, classifyPieces, generateFen
from flask import Flask, request, jsonify
import numpy as np
import cv2

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)


@app.route("/getchessboard", methods=["GET", "POST"])
def get_chessboard():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            # Get the image file
            image_file = request.files["file"]
            # Read the image file using cv2
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # Perform chessboard detection
            found, a, b = find_chessboard(image)
            if found:
                return jsonify({"success": True, "a": a.tolist(), "b": b.tolist()}), 200
            else:
                return (
                    jsonify({"success": False, "message": "No chessboard found"}),
                    404,
                )
        else:
            return jsonify({"error": "File type not allowed"}), 400

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


@app.route("/getfen", methods=["GET", "POST"])
def get_fen():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Get the image file
        image_file = request.files["file"]
        # Read the image file using cv2
        image_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Perform chessboard detection
        found, xl, yl = find_chessboard(image)
        if not found:
            return (
                jsonify({"success": False, "message": "No chessboard found"}),
                404,
            )

        splitBoard(image, xl, yl)
        pieces = classifyPieces()
        return jsonify({"success": True, "fen": generateFen(pieces)}), 200

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
