import numpy as np 
import cv2
import argparse

ap = argparse.ArgumentParser()

# ap.add_argument("-i", "--image", required =True, help="path to input image")
ap.add_argument("-p", "--prototxt", required = True, help="path to model architecture")
ap.add_argument("-m", "--model", required=True, help="model weights file")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum confidence to filter")

args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
cap = cv2.VideoCapture(0)

while cap.isOpened():
	_, image = cap.read()
	# image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)

	detections = net.forward()

	print(detections)

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			text = "{:.2f}%".format(confidence *100)
			y = startY - 10 if startY -10> 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


	cv2.imshow("Output", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
