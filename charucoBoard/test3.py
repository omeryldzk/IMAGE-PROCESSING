# import cv2 library
import cv2

windowname = "video"
cv2.namedWindow(windowname)

videoFilePath = "video.mp4"

capture = cv2.VideoCapture(videoFilePath)

while (capture.isOpened()):

    flag, frame = capture.read()

    if flag:
        cv2.imshow(windowname, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # because 33 * FPS == 1 second
            break
    else:
        break

cv2.destroyWindow(windowname)
capture.release() 
