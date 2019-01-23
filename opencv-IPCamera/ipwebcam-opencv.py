'''
	Reference :
	http://answers.opencv.org/question/133/how-do-i-access-an-ip-camera/

	if above refernce dies : 
	https://web.archive.org/web/20171023194808/http://answers.opencv.org/question/133/how-do-i-access-an-ip-camera/

	Accessing an IP Camera in opencv
	This captures IP Camera, decodes using ffmpeg(opencv must have been built with opencv support) 
	and displays like a normal videofeed 

	The IP Camera is running on my android Phone using 'IP Webcam' App
	It is currently using https over local network, so machines on same network can see the videofeed
	
	IP camera full link reference :
	"http://<username:password>@<ip_address>/video.cgi?.mjpg"
	"http://192.168.226.101:8080/video?x.mjpeg"

	Here, replace the ip address (192.168.226.101:8080) in
	cap.open("http://192.168.226.101:8080/video?x.mjpeg");
	with your ip address !! But make sure you have, ?x.mjpeg ,in the end (where x can be any string).
	If this doesn't work then you should find out what your full ip address is? 
	i.e. find out what is the string after the ip (In my case it was "/video" as shown in above address)

	Using 'colorama' for ANSI escape character sequence (colored terminal text and cursor positioning)
	E.g. 
	# FOREGROUND:
	ESC [ 30 m      # black
	# BACKGROUND
	ESC [ 43 m      # yellow
'''

import numpy as np
import cv2
from colorama import Fore, Back, Style


cap = cv2.VideoCapture()
cap.open("https://192.168.1.2:8080/video?x.mjpeg")

if not (cap.isOpened()):
	print(Back.BLACK+Style.BRIGHT+Fore.YELLOW+"Error : Improvise, adapt, overcome"+Style.RESET_ALL)
	cap.release()

else:
	while (True):
		ret, frame = cap.read()
		if not ret:
			print(Back.BLACK+Style.BRIGHT+Fore.YELLOW+"Error : Cannot read Frame"+Style.RESET_ALL)
			break
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
