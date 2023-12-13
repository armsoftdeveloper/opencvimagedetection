import cv2

img = cv2.imread("images/image.jpg")

full_body = cv2.CascadeClassifier('xml/full_body.xml')

results = full_body.detectMultiScale(img , scaleFactor=1.01 , minNeighbors=1)

for (x , y , w , h) in results:
    cv2.rectangle(img , (x , y),(x + w , y + h), (0,255,0),thickness=2) # thickness hastutuny ramki

cv2.imshow("Result",img)
cv2.waitKey(0)