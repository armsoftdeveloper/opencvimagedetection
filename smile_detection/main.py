import cv2

img = cv2.imread("images/image.jpg")

smile = cv2.CascadeClassifier("xml/smile.xml")

result = smile.detectMultiScale(img , scaleFactor=15.01 , minNeighbors=1)

for (x , y , w , h) in result:
    cv2.rectangle(img , (x , y) , (x + w , y + h), (0,255,0),thickness=2)

cv2.imshow("Result" , img)
cv2.waitKey(0)
cv2.destroyAllWindows()