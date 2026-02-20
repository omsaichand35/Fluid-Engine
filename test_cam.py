import cv2
cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Got Frame:", frame is not None)
cap.release()
