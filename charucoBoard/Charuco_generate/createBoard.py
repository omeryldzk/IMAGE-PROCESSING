import cv2
import cv2.aruco as aruco
import numpy as np

# Define ChArUco board parameters
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
squaresX = 5
squaresY = 7
squareLength = 0.04
markerLength = 0.02
board = aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, ARUCO_DICT)

# Define camera parameters
camera_matrix = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
image_size = (988, 1400)

margin_size = 10  # Adjust the margin size as needed
border_bits = 1  # Adjust the number of border bits as needed
# Generate ChArUco board images from different angles
num_images = 10  # Number of images to capture
output_dir = 'charuco_images/'  # Directory to save the images

for i in range(num_images):
    # Generate a random camera pose (position and orientation)
    rvec, _ = cv2.Rodrigues(np.random.rand(3, 1))  # Random rotation vector
    tvec = np.random.rand(3, 1)  # Random translation vector

    # Generate an image of the ChArUco board from the current camera pose
    image = aruco.drawPlanarBoard(board, image_size, margin_size, border_bits)
    corners, ids, _ = aruco.detectMarkers(image, ARUCO_DICT)
    if len(corners) > 0:
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, image, board
        )
        if retval > 0:
            aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

            # Draw axes on the image
            axis_length = markerLength * 2
            rvec_matrix, _ = cv2.Rodrigues(rvec)
            axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
            image_points, _ = cv2.projectPoints(axis_points, rvec_matrix, tvec, camera_matrix, dist_coeffs)
            # Reshape the image_points to be in the correct format
            image_points = np.squeeze(image_points)
            image_points = np.round(image_points).astype(int)

            # Define the points for the axes
            pt1 = tuple(image_points[0])
            pt2 = tuple(image_points[1])
            pt3 = tuple(image_points[2])
            pt4 = tuple(image_points[3])

            # Draw lines representing the axes
            image = cv2.line(image, pt1, pt2, (0, 0, 255), 2)  # X-axis (red)
            image = cv2.line(image, pt1, pt3, (0, 255, 0), 2)  # Y-axis (green)
            image = cv2.line(image, pt1, pt4, (255, 0, 0), 2)  # Z-axis (blue)


    image_path = f'charuco_{i+20}.jpg'
    cv2.imwrite(image_path, image)

    # Display the image (optional)
    cv2.imshow('ChArUco Image', image)
    cv2.waitKey(500)  # Pause for 500 milliseconds

cv2.destroyAllWindows()
