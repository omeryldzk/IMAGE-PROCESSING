import cv2
import cv2.aruco as aruco
import numpy as np

# Create ChArUco board, which is a set of Aruco markers in a chessboard setting meant for calibration
# The following call gets a ChArUco board of tiles 5 wide X 7 tall
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

gridboard = aruco.CharucoBoard(
    (5,
    7),
    squareLength=0.04,
    markerLength=0.02,
    dictionary=ARUCO_DICT
)

# Create an image from the gridboard
image_size = (988, 1400)

margin_size = 10  # Adjust the margin size as needed
border_bits = 1  # Adjust the number of border bits as needed
board_img = aruco.drawPlanarBoard(gridboard, image_size, margin_size, border_bits)



cv2.imwrite("test_charuco_1.jpg", board_img)

# Display the image
cv2.imshow('Gridboard', board_img)
# Exit on any key
cv2.waitKey(500)
cv2.destroyAllWindows()
