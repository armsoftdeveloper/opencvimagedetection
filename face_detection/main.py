import cv2

img = cv2.imread("images/image.jpg")

faces = cv2.CascadeClassifier('xml/faces.xml')

results = faces.detectMultiScale(img , scaleFactor=1.01 , minNeighbors=3)

for (x , y, w ,h) in results:
    cv2.rectangle(img , (x , y),(x + w , y + h), (0,255,0),thickness=2) # thickness hastutuny ramki

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result" , 1000 , 1000)
cv2.imshow("Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

