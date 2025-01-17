
import numpy as np
import cv2
from cv2 import aruco
import glob


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard(
        (CHARUCOBOARD_COLCOUNT,
        CHARUCOBOARD_ROWCOUNT),
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)
#///////////////////// board fotolarını üretme /////////////////
# Kendim camera matrix değerlerini verdim
# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime


images = glob.glob('./charuco_*.jpeg')
counter = 0

for iname in images:
    # Open the image
    img = cv2.imread(iname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    img = aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

    # Get charuco corners and ids from detected aruco markers
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    if response > 10:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
       
        # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]
    
        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
        # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        cv2.waitKey(5)
        counter+=1
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()


# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)



retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, CHARUCO_BOARD, cameraMatrix, distCoeffs,np.zeros((3, 1)),
    np.zeros((3, 1)))  # posture estimation from a charuco board

rotM, _ = cv2.Rodrigues(rvec)
camera_position = -np.dot(rotM.T, tvec)

print("Camera Position (x, y, z):", camera_position)


# Convert the 3D coordinates to a numpy array
chessboard_3d_corners = np.array(CHARUCO_BOARD.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)

# Project the 3D corners onto the image plane
imagePoints, _ = cv2.projectPoints(chessboard_3d_corners, rvec, tvec, cameraMatrix, distCoeffs)

## Select a particular image point (for example, the first one)
selected_image_point = imagePoints[0][0].astype(int)

# Convert the selected image point to a homogeneous coordinate (3x1)
selected_image_point_homogeneous = np.array([selected_image_point[0], selected_image_point[1], 1], dtype=np.float32).reshape(-1, 1)

# Back-project the selected image point to 3D world coordinates
intrinsic_matrix_inv = np.linalg.inv(cameraMatrix[:3, :3])  # Take the 3x3 intrinsic matrix
world_point_homogeneous = np.dot(intrinsic_matrix_inv, selected_image_point_homogeneous)

# Normalize the 3D world point by dividing by its third element (z)
world_point_3d = world_point_homogeneous[:2] / world_point_homogeneous[2]

print("3D Position of Camera's Projection on the ChArUco Board:", world_point_3d)
