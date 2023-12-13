import cv2

img = cv2.imread("images/image.jpg")

plate_number =cv2.CascadeClassifier("xml/plate.xml")

result = plate_number.detectMultiScale(img , scaleFactor=1.01 , minNeighbors=3)

for (x , y , w , h) in result:
    cv2.rectangle(img , (x , y) , (x + w , y + h) , (0,255,0) , thickness=3)

cv2.imshow("Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()