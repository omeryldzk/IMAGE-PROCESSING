#include <iostream>

#include <eigen3/Eigen/Dense>


bool computePixelCoordinates(
    Eigen::Vector3f &pWorld,
    Eigen::Matrix4f &cameraToWorld,
    const float &canvasWidth,
    const float &canvasHeight,
    const int &imageWidth,
    const int &imageHeight,
    Eigen::Vector2i &pRaster)
{
    // First, transform the 3D point from world space to camera space.
    // It is, of course inefficient to compute the inverse of the cameraToWorld
    // matrix in this function. It should be done only once outside the function
    // and the worldToCamera should be passed to the function instead.
    // We only compute the inverse of this matrix in this function ...
    Eigen::Vector3f pCamera;
    Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();
    //
    pCamera.x() = worldToCamera.coeff(0, 0) * pWorld.x() + worldToCamera.coeff(1, 0) * pWorld.y()
    + worldToCamera.coeff(2, 0) * pWorld.z() + worldToCamera.coeff(3, 0);
    pCamera.y() = worldToCamera.coeff(0, 1) * pWorld.x() + worldToCamera.coeff(1, 1) * pWorld.y()
    + worldToCamera.coeff(2, 1) * pWorld.z() + worldToCamera.coeff(3, 1);
    pCamera.z() = worldToCamera.coeff(0, 2) * pWorld.x() + worldToCamera.coeff(1, 2) * pWorld.y()
    + worldToCamera.coeff(2, 2) * pWorld.z() + worldToCamera.coeff(3, 2);
    // Coordinates of the point on the canvas. Use perspective projection.
    std::cout << "1 : "<< pCamera << std::endl;
    Eigen::Vector2f pScreen;
    pScreen.x() = pCamera.x() / -pCamera.z();
    pScreen.y() = pCamera.y() / -pCamera.z();
    std::cout << "2 : "<< pScreen << std::endl;
    // If the x- or y-coordinate absolute value is greater than the canvas width
    // or height respectively, the point is not visible
    if (std::abs(pScreen.x()) > canvasWidth || std::abs(pScreen.y()) > canvasHeight)
            return false;
    // Normalize. Coordinates will be in the range [0,1]
    Eigen::Vector2f pNDC;
    pNDC.x() = (pScreen.x() + (canvasWidth/ 2))  / canvasWidth;
    pNDC.y() = (pScreen.y() + (canvasHeight/ 2)) / canvasHeight;
    std::cout << pNDC << std::endl;
    // Finally, convert to pixel coordinates. Don't forget to invert the y coordinate
    pRaster.x() = std::floor(pNDC.x() * imageWidth);
    pRaster.y() = std::floor((1 - pNDC.y()) * imageHeight);

    return true;
}
int main()
{
    Eigen::Matrix4f cameraToWorld;
    cameraToWorld << 0.718762, 0.615033, -0.324214, 0, -0.393732, 0.744416, 0.539277, 0, 0.573024, -0.259959, 0.777216, 0, 0.526967, 1.254234, -2.53215, 1;
    Eigen::Vector3f pWorld(-0.315792, 1.4489, -2.48901);
    float canvasWidth = 2, canvasHeight = 2;
    uint32_t imageWidth = 512, imageHeight = 512;
    // The 2D pixel coordinates of pWorld in the image if the point is visible
    Eigen::Vector2i pRaster;
    if (computePixelCoordinates(pWorld, cameraToWorld, canvasWidth, canvasHeight, imageWidth, imageHeight, pRaster)) {
            std::cout << "Pixel coordinates :" << pRaster.x()<< " : "<< pRaster.y()<< std::endl;
        }
    else {
            std::cout << pWorld << " is not visible" << std::endl;
        }
    return 0;
}
