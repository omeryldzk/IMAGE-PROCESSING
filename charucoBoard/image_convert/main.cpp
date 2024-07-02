#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;

int main()
{
    // ChAruco board variables
    int CHARUCOBOARD_ROWCOUNT = 7;
    int CHARUCOBOARD_COLCOUNT = 5;
    Ptr<aruco::Dictionary> ARUCO_DICT = aruco::getPredefinedDictionary(aruco::DICT_5X5_1000);

    // Create the Charuco board
    Ptr<aruco::CharucoBoard> CHARUCO_BOARD = aruco::CharucoBoard::create(CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT, 0.04f, 0.02f, ARUCO_DICT);

    // Arrays to store Charuco corners and IDs from images processed
    vector<vector<Point2f>> corners_all;
    vector<vector<int>> ids_all;

    // Determine image size at runtime
    Size image_size;

    vector<String> images;
    glob("./charuco_*.jpeg", images);

    for (const auto& iname : images)
    {
        // Open the image
        Mat img = imread(iname);
        // Grayscale the image
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // Find aruco markers in the query image
        vector<vector<Point2f>> corners;
        vector<int> ids;
        aruco::detectMarkers(gray, ARUCO_DICT, corners, ids);

        // Get charuco corners and ids from detected aruco markers
        vector<Point2f> charuco_corners;
        vector<int> charuco_ids;
        int response = aruco::interpolateCornersCharuco(corners, ids, gray, CHARUCO_BOARD, charuco_corners, charuco_ids);

        // If a Charuco board was found, let's collect image/corner points
        // Requiring at least 20 squares
        if (response > 10)
        {
            // Add these corners and ids to our calibration arrays
            corners_all.push_back(charuco_corners);
            ids_all.push_back(charuco_ids);

            // If our image size is unknown, set it now
            if (image_size.width == 0 && image_size.height == 0)
            {
                image_size = gray.size();
            }

            // Draw the Charuco board we've detected to show our calibrator the board was properly detected
            aruco::drawDetectedCornersCharuco(img, charuco_corners, charuco_ids);

            // Reproportion the image, maxing width or height at 1000
            double proportion = max(img.size().width, img.size().height) / 1000.0;
            resize(img, img, Size(static_cast<int>(img.size().width / proportion), static_cast<int>(img.size().height / proportion)));

            // Display the image
            imshow("Charuco board", img);
            waitKey(5);
        }
        else
        {
            cout << "Not able to detect a charuco board in image: " << iname << endl;
        }
    }

    // Make sure at least one image was found
    if (images.empty())
    {
        cout << "Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file." << endl;
        return 1;
    }

    // Now that we've seen all of our images, perform the camera calibration
    // based on the set of points we've discovered
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    double calibrationError = aruco::calibrateCameraCharuco(corners_all, ids_all, CHARUCO_BOARD, image_size, cameraMatrix, distCoeffs, rvecs, tvecs);

    // Print camera matrix and distortion coefficients
    cout << "Camera Matrix:" << endl;
    cout << cameraMatrix << endl;
    cout << "Distortion Coefficients:" << endl;
    cout << distCoeffs << endl;

    // Estimate camera pose for the last image
    Mat rvec, tvec;
    bool poseEstimationSuccess = aruco::estimatePoseCharucoBoard(corners_all.back(), ids_all.back(), CHARUCO_BOARD, cameraMatrix, distCoeffs, rvec, tvec);

    if (poseEstimationSuccess)
    {
        Mat rotM;
        Rodrigues(rvec, rotM);
        Mat camera_position = -rotM.t() * tvec;

        cout << "Camera Position (x, y, z):" << endl;
        cout << camera_position << endl;
    }
    else
    {
        cout << "Pose estimation for the last image failed." << endl;
    }

    // Convert the 3D coordinates to a vector of Point3f
    vector<Point3f> chessboard_3d_corners;
    for (int i = 0; i < CHARUCOBOARD_ROWCOUNT; i++)
    {
        for (int j = 0; j < CHARUCOBOARD_COLCOUNT; j++)
        {
            chessboard_3d_corners.push_back(Point3f(j * 0.04f, i * 0.04f, 0));
        }
    }

    // Project the 3D corners onto the image plane
    vector<Point2f> projected_points;
    projectPoints(chessboard_3d_corners, rvec, tvec, cameraMatrix, distCoeffs, projected_points);

    // Select a particular image point (for example, the first one)
    Point2f selected_image_point = projected_points[0];

    // Back-project the selected image point to 3D world coordinates
    Mat intrinsic_matrix_inv = cameraMatrix.inv(DECOMP_LU);
    Mat selected_image_point_homogeneous = (Mat_<float>(3, 1) << selected_image_point.x, selected_image_point.y, 1);
    Mat world_point_homogeneous = intrinsic_matrix_inv * selected_image_point_homogeneous;

    // Normalize the 3D world point by dividing by its third element (z)
    Point2f world_point_3d(world_point_homogeneous.at<float>(0) / world_point_homogeneous.at<float>(2),
                           world_point_homogeneous.at<float>(1) / world_point_homogeneous.at<float>(2));

    cout << "3D Position of Camera's Projection on the ChArUco Board:" << endl;
    cout << world_point_3d << endl;

    return 0;
}
