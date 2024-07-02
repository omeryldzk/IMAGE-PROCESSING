from memory_profiler import profile
import numpy as np
import cv2
import cv2.aruco as aruco
@profile
def calibrate_camera(video_file, charuco_board,dict,camera_matrix,dist_coeffs):
   
    # Arrays to store object points and image points from all frames
    obj_points = []  
    img_points = []  
    

    # Capture video frames
    cap = cv2.VideoCapture(video_file)
    charuco_corners = None
    charuco_ids = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for Charuco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Charuco board corners and ids
        corners, ids, _ = aruco.detectMarkers(gray, dict)
        if ids is not None:
            charuco_response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
            if charuco_response is not None and charuco_response > 10:
                img_points.append(charuco_corners)
                obj_points.append(charuco_ids)
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

    _, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs,np.zeros((3, 1)),
    np.zeros((3, 1)))  # posture estimation from a charuco board

    return rvec, tvec

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
    rvec ,tvec= calibrate_camera(video_file, charuco_board,charuco_dict,camera_matrix,dist_coeffs)
    #
    rotM, _ = cv2.Rodrigues(rvec)
    camera_position = -np.dot(rotM.T, tvec)

    


    # Convert the 3D coordinates to a numpy array
    chessboard_3d_corners = np.array(charuco_board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)

    # Project the 3D corners onto the image plane
    imagePoints, _ = cv2.projectPoints(chessboard_3d_corners, rvec, tvec, camera_matrix, dist_coeffs)
    
    ## Select a particular image point (for example, the first one)
    selected_image_point = imagePoints[5][0].astype(int)
    print(selected_image_point)
    # Convert the selected image point to a homogeneous coordinate (3x1)
    selected_image_point_homogeneous = np.array([selected_image_point[0], selected_image_point[1], 1], dtype=np.float32).reshape(-1, 1)

    # Back-project the selected image point to 3D world coordinates
    intrinsic_matrix_inv = np.linalg.inv(camera_matrix[:3, :3])  # Take the 3x3 intrinsic matrix
    world_point_homogeneous = np.dot(intrinsic_matrix_inv, selected_image_point_homogeneous)

    # Normalize the 3D world point by dividing by its third element (z)
    world_point_3d = world_point_homogeneous[:2] / world_point_homogeneous[2]

    print("3D Position of Camera's Projection on the ChArUco Board:", world_point_3d)
