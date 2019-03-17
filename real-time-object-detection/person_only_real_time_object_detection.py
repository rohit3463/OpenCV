# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
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

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream('1.mp4').start()
time.sleep(2.0)
fps = FPS().start()

#upper and lower bound for yellow color
lowerBound=np.array([20,100,100])
upperBound=np.array([30,255,255])

#morphological kernels
kernelOpen=np.ones((13,13))
kernelClose=np.ones((20,20))

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	if frame is None:
		continue
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
		if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			#show bounding box only on person
			if CLASSES[idx] == "person":
				# draw the prediction on the frame
				startX = max([0, startX])
				startY = max([0, startY])
				endX = min([frame.shape[1], endX])
				endY = min([frame.shape[0], endY])
				
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

					#area = cv2.contourArea(max_cont)
					#print(area)
					if True:

						x,y,wi,hi=cv2.boundingRect(max_cont)
						cv2.rectangle(person,(x,y),(x+wi,y+hi),(0,0,255), 2)
						
						(pw, ph, pc) = person.shape
						pw = startX + int(pw/2)
						ph = startY + int(ph/2)

						#cv2.circle(frame, (pw, ph), 10, (0,0,255))
						
						frame[startY:endY, startX:endX] = person

	# show the output frameimgimgimg
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()