from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

#upper and lower bound for yellow color
lowerBound = np.array([110,50,50])
upperBound = np.array([130,255,255])

#morphological kernels
kernelOpen = np.ones((3,3))
kernelClose = np.ones((20,20))

# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
frame = cv2.imread('test.jpeg')
frame = imutils.resize(frame, width=400)

# grab the frame dimensions and convert it to a blob
(h, w, c) = frame.shape
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > 0.3:
        # extract the index of the class label from the
        # `detections`, then compute the (x, y)-coordinates of
        # the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        #show bounding box only on person
        if CLASSES[idx] == "person":
            # draw the prediction on the frame

            '''
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            '''
            
            startX = max([0, startX])
            startY = max([0, startY])
            endX = min([w-1, endX])
            endY = min([h, endY])
            
            person = frame[startY:endY, startX:endX]
            cv2.imshow("person" ,person)

            imgHSV = cv2.cvtColor(person,cv2.COLOR_BGR2HSV)

            # create the Mask
            mask = cv2.inRange(imgHSV,lowerBound,upperBound)
            cv2.imshow('mask', mask)

            #morphology
            maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
            maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

            maskFinal=maskClose
            im2,conts,hg=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            if len(conts) != 0:
                max_cont = max(conts, key=cv2.contourArea)

                x,y,wi,hi=cv2.boundingRect(max_cont)
                cv2.rectangle(person,(x,y),(x+wi,y+hi),(0,0,255), 2)
                
                frame[startY:endY, startX:endX] = person

    # show the output frameimgimgimg
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)


# do a bit of cleanup
cv2.destroyAllWindows()
