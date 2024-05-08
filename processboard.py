import cv2
from ultralytics import YOLO

model = YOLO("best.pt", task="predict", verbose=False)


def classifyPieces():
    pieces = []
    results = model("temp-board")
    for r in results:
        id = r.probs.top1
        piece = r.names[id]
        pieces.append(piece)

    assert len(pieces) == 64, "Number of images must be 64"

    return pieces


def generateFen(pieces):
    result = ""
    blanks = 0
    for i, piece in enumerate(pieces):
        if piece == "empty":
            blanks += 1
        else:
            if blanks != 0:
                result += str(blanks)
                blanks = 0
            result += piece[1] if piece[0] == "b" else piece[1].upper()
        if (i + 1) % 8 == 0:
            if blanks != 0:
                result += str(blanks)
                blanks = 0
            result += "/"
    return result


# img = cv2.imread("classify-pieces/pieces/b-r.png")
# img = cv2.imread("test-images/respawn.png")
# ok, xl, yl = find_chessboard(img)
# splitBoard(img, xl, yl)
# pieces = classifyPieces()
# print(generateFen(pieces))
