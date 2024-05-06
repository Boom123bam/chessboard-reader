import cv2
from ultralytics import YOLO
from findchessboard import find_chessboard

model = YOLO("flask-api/best.pt", task="predict", verbose=False)


def classifyPieces():
    pieces = []
    results = model("temp-board")
    for r in results:
        id = r.probs.top1
        piece = r.names[id]
        pieces.append(piece)

    return pieces


def splitBoard(boardImg, xLines, yLines):
    w, h, _ = boardImg.shape
    # tiles = []

    i = 0
    for x in range(8):
        for y in range(8):
            x1 = round(xLines[x] * w)
            x2 = round(xLines[x + 1] * w)
            y1 = round(yLines[y] * h)
            y2 = round(yLines[y + 1] * h)
            cv2.imwrite(f"temp-board/{'{:02d}'.format(i)}.png", boardImg[x1:x2, y1:y2])
            # tiles.append(boardImg[x1:x2, y1:y2])
            i += 1


# img = cv2.imread("classify-pieces/pieces/b-r.png")
# img = cv2.imread("test-images/respawn.png")
# ok, xl, yl = find_chessboard(img)
# splitBoard(img, xl, yl)
classifyPieces()
