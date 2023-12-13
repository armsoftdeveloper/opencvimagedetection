import cv2

img = cv2.imread('images/image.jpg')

eye_tree = cv2.CascadeClassifier('xml/haarcascade_eye_tree_eyeglasses.xml')

results = eye_tree.detectMultiScale(img , scaleFactor=1.01 , minNeighbors=3)

for (x , y , w , h) in results:
    cv2.rectangle(img , (x , y),(x + w , y + h), (0,255,0),thickness=3) # thickness hastutuny ramki

cv2.imshow("Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()