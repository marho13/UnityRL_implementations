import cv2
import os

print(os.listdir("C:\\Users\\Martin\\Desktop"))
img = cv2.imread("C:\\Users\\Martin\\Desktop\\250.png", -1)
print(len(img), len(img[0]), len(img[0][0]))
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# gbr = img[..., [2, 0, 1]].copy()
cv2.imshow("150", img)
cv2.waitKey(0)