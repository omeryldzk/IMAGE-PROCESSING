import cv2
import numpy as np
import cv2.aruco as aruco
import glob

class CameraCalibrator:
    def __init__(self, charuco_dict, charuco_board, mtx=None, dist=None):
        self.mtx = mtx
        self.dist = dist
        self.charuco_dict = charuco_dict
        self.charuco_board = charuco_board

    def estimate_pose(self, image_names):
        # Loop over all images
        for image_name in image_names:
            # Load image
            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect Charuco board corners and ids
            corners, ids, _ = aruco.detectMarkers(gray, self.charuco_dict)
            charuco_response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)

            if charuco_response is not None and charuco_response > 10:
                # Use solve PnP to determine the rotation and translation between camera and 3D object
                _, rvec, tvec = cv2.solvePnP(self.charuco_board.getChessboardCorners(), charuco_corners, self.mtx, self.dist)

                # Project the axis into the image
                axis = np.float32([[0.04, 0, 0], [0, 0.04, 0], [0, 0, -0.04]]).reshape(-1, 3)
                imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.mtx, self.dist)

                # Draw the axes
                img = self.draw_axes(img, charuco_corners.squeeze(), imgpts.squeeze())

                # Calculate camera position
                rotM = cv2.Rodrigues(rvec)[0]
                cameraPosition = -np.dot(rotM.T, tvec)

                imgpts, _ = cv2.projectPoints(cameraPosition, rvec, tvec, self.mtx, self.dist)
                img = self.draw(img, charuco_corners[0, 0], imgpts[0, 0])

            cv2.imshow('img', img)
            k = cv2.waitKey(0) & 0xFF

    def draw_axes(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(len(imgpts)):
            tmp = tuple(imgpts[i].ravel())
            tmp = (int(tmp[0]), int(tmp[1]))
            img = cv2.line(img, corner, tmp, color[i], 5)
        return img

    def draw(self, img, corners, imgpts):
        corner = tuple(corners.ravel())
        corner = (int(corner[0]), int(corner[1]))
        for i in range(len(imgpts)):
            tmp = tuple(imgpts[i].ravel())
            tmp = (int(tmp[0]), int(tmp[1]))
            img = cv2.line(img, corner, tmp, (255, 255, 0), 5)
        cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), 1, (255, 255, 255), 10)
        return img

def main():
    images = glob.glob('./charuco_*.jpeg')
    # Load the Charuco board parameters
    charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    
    charuco_board = aruco.CharucoBoard((5, 7), 0.04, 0.02, charuco_dict)

    # Precalibrated camera information
    mtx = np.array([[1.24716235e+03, 0.00000000e+00, 8.03972796e+02],
                              [0.00000000e+00, 1.25489224e+03, 2.61385258e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array([[0.15167277, -0.23128832, -0.0372587, 0.00291309, 0.17613692]])

    camera_calibrator = CameraCalibrator(charuco_dict, charuco_board, mtx=mtx, dist=dist)
    camera_calibrator.estimate_pose(images)


if __name__ == '__main__':
    main()
