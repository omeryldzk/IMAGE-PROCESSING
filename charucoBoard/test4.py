import math
import numpy as np
import cv2
import cv2.aruco as aruco
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
def calibrate_camera(video_file, charuco_board,dict):
   
    # Arrays to store object points and image points from all frames
    all_corners = []  
    all_ids = []  
    

    # Capture video frames
    cap = cv2.VideoCapture(video_file)
    charuco_corners = None
    charuco_ids = None
    counter = 0
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        counter += 1
        if counter % 60 != 0:
            continue 
        # Convert frame to grayscale for Charuco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Charuco board corners and ids
        corners, ids, _ = aruco.detectMarkers(gray, dict)
        if ids is not None:
            charuco_response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
            if charuco_response is not None and charuco_response > 10:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
            # Draw detected markers and Charuco board on the frame
                frame = aruco.drawDetectedMarkers(frame, corners)
                if charuco_corners is not None and charuco_ids is not None:
                    frame = aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) <= 3:
        print("Not enough valid Charuco corners detected for calibration.")
        return None, None

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco( charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None)
    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, mtx, dist,np.zeros((3, 1)),
np.zeros((3, 1)))  # posture estimation of charuco board
    rotM, _ = cv2.Rodrigues(rvec)
    print('Translation : {0}'.format(tvec))
    print('Rotation    : {0}'.format(rotationMatrixToEulerAngles(rotM)))
    print('Distance from camera: {0} m'.format(np.linalg.norm(tvec)))
    return rvec,tvec,rvecs, tvecs,mtx,dist

def get_camera_pos(rvecs,tvecs,camera_matrix,dist_coeffs):
    for rvec,tvec in zip(rvecs,tvecs):
        rotM, _ = cv2.Rodrigues(rvec)
        camera_position = -np.dot(rotM.T, tvec)
        print("Camera Position (x, y, z):", camera_position)
        
if __name__ == '__main__':
    # Define the Charuco board parameters
    charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    
    charuco_board = aruco.CharucoBoard((5, 7), 0.04, 0.02, charuco_dict)

    # Set the size of the squares on the Charuco board (in arbitrary units)
    square_size = 0.04

    # Set the size of the Aruco markers on the Charuco board (in arbitrary units)
    marker_size = 0.02

    # Provide the path to the video file
    video_file = 'video.mp4'

    # Perform camera calibration
    rvec,tvec,rvecs ,tvecs,mtx,dist= calibrate_camera(video_file, charuco_board,charuco_dict)
    #find camera pos
    get_camera_pos(rvecs,tvecs,mtx,dist)
    