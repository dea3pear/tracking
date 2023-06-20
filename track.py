
import cv2

webcam=cv2.VideoCapture(0)

object_detection = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold=35)

while webcam.isOpened:
    ret,frame=webcam.read()

    mask = object_detection.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(frame,[cnt], -1, (0, 255, 0), 2)
            x, y, w, h, = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 3)

    if ret==True:
        cv2.imshow("mycam",frame)
        cv2.imshow("camask",mask)
        key=cv2.waitKey(1)
        if key==ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()