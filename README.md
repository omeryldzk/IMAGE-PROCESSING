# Robotic Arm Pose Estimation and Camera Calibration

## Overview
During my two-month internship, I worked on addressing a robotic arm problem by mastering and applying advanced algorithms in camera calibration and pose estimation. This project involved:

- Implementing and optimizing camera calibration and pose estimation algorithms using Python.
- Applying image processing techniques to enhance the performance of these algorithms.
- Translating the Python code into C++ for improved efficiency and integration.
- Exploring pose estimation with point clouds and introducing the high-performance Pose Lookup Method as an alternative to traditional methods.

## Key Components

### Camera Calibration
Camera calibration is crucial for accurate pose estimation. During the project, I utilized various techniques to achieve precise calibration, ensuring reliable results for the robotic arm's movements.

### Pose Estimation
Pose estimation was tackled using two main approaches:
1. **Traditional Methods:** Initially, I implemented traditional pose estimation methods to establish a baseline performance.
2. **Pose Lookup Method:** In the second month, I introduced the Pose Lookup Method, which significantly improved the performance and accuracy of the pose estimation process.

### Image Processing Techniques
To enhance the algorithm's performance, I applied several image processing techniques, including filtering, edge detection, and feature extraction. These techniques helped in refining the input data, leading to more accurate pose estimations.

### Code Translation to C++
To ensure high performance and seamless integration with existing systems, I translated the Python code into C++. This step was crucial for deploying the algorithms in a real-world robotic arm application.

## Technologies Used
- **Programming Languages:** Python, C++
- **Libraries and Tools:** OpenCV, NumPy, PCL (Point Cloud Library), Eigen
- **Concepts:** Camera Calibration, Pose Estimation, Image Processing, Point Clouds

## Learning and Growth
This project was a significant learning experience, enriching my knowledge and expertise in AI technologies, especially in the fields of image processing and pose estimation. Staying abreast of the latest advancements in these areas was key to the success of this project.

## Repository Structure
- `charucoBoard/`: Contains the source code for the charucoBoard files, including Python and C++ implementations.
- `charucoBoard/image_convert`: Includes photos of charucoBoard at diffrent angles and C and python source codes for calibrating and converting coordinates between real life and image.
- `charucoBoard/video_convert`: Includes video of charucoBoard and python source code for calibrating and converting coordinates between real life and image.
- `charucoBoard/generate`: Includes python source code for generating charucoBoard.
- `matrix/`: Contains the source code for the image processing algorithms which is rotation matrixs and PNP method, including Python and C++ implementations.

## Getting Started
To get started with this project, clone the repository and follow the instructions in the `docs/` directory.

```bash
git clone https://github.com/yourusername/robotic-arm-pose-estimation.git
cd robotic-arm-pose-estimation
```
