import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):

    # Take each frame
    ret, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([0,86,6])
    upper_red = np.array([14,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Erode and dilate the mask to remove noise and smooth it
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if this check is not in place the prorgam crashes when no object is found
    if len(contours)>0:
        cnt = max(contours,key=cv2.contourArea)
        # draw min-enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    
    # exit of keypress 'q'
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# release video camera and destroy all windows
cap.release()
cv2.destroyAllWindows()