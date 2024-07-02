from memory_profiler import profile
import numpy as np
import cv2
import cv2.aruco as aruco
@profile
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
        if counter % 20 != 0:
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
np.zeros((3, 1)))  # posture estimation from a charuco board
    return rvec, tvec, mtx, dist
def get_camera_pos(rvec,tvec,mtx,dist):
    # Calculate camera position. Following: https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp?rq=1
    rotM, _ = cv2.Rodrigues(rvec)
    camera_position = -np.dot(rotM.T, tvec)
    
    imgpts, jac = cv2.projectPoints(camera_position, rvec, tvec, mtx, dist)
    
def draw_axes(img, corners, imgpts):
        # Extract the first corner (the top left)
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))

        # Color format is BGR
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        # Iterate over the points
        for i in range(len(imgpts)):
            tmp = tuple(imgpts[i].ravel())
            tmp = (int(tmp[0]), int(tmp[1]))
            img = cv2.line(img, corner, tmp, color[i], 5)
        return img

def draw( img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = (int(corner[0]), int(corner[1]))
    for i in range(len(imgpts)):
        tmp = tuple(imgpts[i].ravel())
        tmp = (int(tmp[0]), int(tmp[1]))
        img = cv2.line(img, corner, tmp, (255, 255, 0), 5)
    cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), 1, (255, 255, 255), 10)
    return img

if __name__ == '__main__':
    # Define the Charuco board parameters
    camera_matrix = np.array([[1.24716235e+03, 0.00000000e+00, 8.03972796e+02],
                              [0.00000000e+00, 1.25489224e+03, 2.61385258e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist_coeffs = np.array([[0.15167277, -0.23128832, -0.0372587, 0.00291309, 0.17613692]])
    
    charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    
    charuco_board = aruco.CharucoBoard((5, 7), 0.04, 0.02, charuco_dict)

    # Set the size of the squares on the Charuco board (in arbitrary units)
    square_size = 0.04

    # Set the size of the Aruco markers on the Charuco board (in arbitrary units)
    marker_size = 0.02

    # Provide the path to the video file
    video_file = 'video.mp4'

    # Perform camera calibration
    rvecs ,tvecs= calibrate_camera(video_file, charuco_board,charuco_dict)
    #find camera pos
    get_camera_pos(rvecs,tvecs,camera_matrix,dist_coeffs)
    