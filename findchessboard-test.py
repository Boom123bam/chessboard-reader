from findchessboard import find_chessboard
import os
import cv2


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


badfolder = "test-images/without-board"
goodfolder = "test-images/with-board"

falsePos = 0
falseNeg = 0
truePos = 0
trueNeg = 0

# Loop over files in the folder
for filename in os.listdir(badfolder):
    file_path = os.path.join(badfolder, filename)
    if not os.path.isfile(file_path) or not allowed_file(file_path):
        continue

    image = cv2.imread(file_path)
    ok, xl, yl = find_chessboard(image)

    if ok:
        print(file_path, f"Expected no chessboard, got: {xl}, {yl}")
        falsePos += 1
    else:
        trueNeg += 1


for filename in os.listdir(goodfolder):
    file_path = os.path.join(goodfolder, filename)
    if not os.path.isfile(file_path) or not allowed_file(file_path):
        continue

    image = cv2.imread(file_path)
    ok, xl, yl = find_chessboard(image)

    if not ok:
        print(file_path, f"Expected chessboard, got none")
        falseNeg += 1
    else:
        truePos += 1

print(
    f"{truePos} true positives | {trueNeg} true negatives | {falsePos} false positives | {falseNeg} false negatives"
)
