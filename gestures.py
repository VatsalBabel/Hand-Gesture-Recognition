#Importings
import cv2
import numpy as np

#Video Capture
cap = cv2.VideoCapture(0)

while True:
	#Frame capture and grayscale conversion
	ret, frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#Detector area
	cv2.rectangle(frame, (100, 100), (250, 250), (0, 255, 0), 4)
	gray_frame = gray_frame[100:250, 100:250]

	#Filters for detecting hand
	ret, threshold = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)
	kernel = np.ones((3, 3), np.uint8)
	threshold = cv2.dilate(threshold, kernel, iterations=2)
	kernel = np.ones((1, 1), np.uint8)
	threshold = cv2.erode(threshold, kernel, iterations=4)
	threshold = cv2.GaussianBlur(threshold, (5, 5), 0)

	#Finding the contours
	_, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	try:
		#Finding the contours with max area and the convex hull
		#Visualization of the hull
		max_area = 0	
		for i in range(len(contours)):
			cnt = contours[i]
			area = cv2.contourArea(cnt)
			if area>max_area:
				max_area = area
				ci = i
		cnt = contours[ci]
		hull = cv2.convexHull(cnt)
		cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
		hull = cv2.convexHull(cnt, returnPoints = False)
		
		#Finding the defects
		defects = cv2.convexityDefects(cnt, hull)

		#Convexity Defect Visualization
		number = defects.shape[0] - 1
		if number == 1:
			cv2.putText(frame, 'ONE', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
		elif number == 2:
			cv2.putText(frame, 'TWO', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
		elif number == 3:
			cv2.putText(frame, 'THREE', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
		elif number == 4:
			cv2.putText(frame, 'FOUR', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
		elif number == 5 or number == 6:
			cv2.putText(frame, 'FIVE', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

		for i in range(defects.shape[0]):
			_ , _, f, _ = defects[i,0]
			far = tuple(cnt[f][0])
			cv2.circle(gray_frame, far, 2, [255, 255, 255], -1)
	except:
		pass

	cv2.imshow('gray_frame', gray_frame)
	cv2.imshow('threshold', threshold)
	cv2.imshow('frame', frame)
	
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cv2.destroyAllWindows()
cap.release()
