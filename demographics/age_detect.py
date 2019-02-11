import numpy as np 
import cv2

face_net = cv2.dnn.readNetFromCaffe("face.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("second/age.prototxt", "second/dex_chalearn_iccv2015.caffemodel")
cap = cv2.VideoCapture(0)

while cap.isOpened():
	_, image = cap.read()
	# image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
	(h, w) = image.shape[:2]

	blob_face = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	face_net.setInput(blob_face)

	detections = face_net.forward()

	# print(detections)

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			y = startY - 10 if startY -10> 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			

			age_startX = int(startX - 0.4 * startX) if int(startX - 0.4 * startX) > 0 else startX
			age_endX = int(endX + 0.4 * endX) if int(endX + 0.4 * endX) < w else endX 
			age_startY = int(startY - 0.4 * startY) if int(startY - 0.4 * startY) > 0 else startY 
			age_endY = int(endY + 0.4 * endY) if int(endY + 0.4 * endY) < h else endY

			blob_age = cv2.dnn.blobFromImage(cv2.resize(image[age_startY:age_endY, age_startX:age_endX], (224, 224)), 1.0, (224, 224), (0.0, 0.0, 0.0), swapRB = False)

			age_net.setInput(blob_age)

			detected_age = age_net.forward()

			mean_softmax = np.sum(detected_age * np.arange(101))

			text = "{:.2f} age".format(mean_softmax) 

			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Output", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
