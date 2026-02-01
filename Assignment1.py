#QUESTION 1.1
import cv2

image = cv2.imread("C:\Users\DHARSHANA D\OneDrive\Pictures\Screenshots\Screenshot 2025-12-20 011807.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edge_image = cv2.Canny(blur_image, 100, 200)

cv2.imshow('Gray', gray_image)
cv2.imshow('Blur', blur_image)
cv2.imshow('Edges', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#QUESTION 1.2
import cv2

img = cv2.imread("C:\Users\DHARSHANA D\OneDrive\Pictures\Screenshots\Screenshot 2025-12-20 011807.png")
h, w = img.shape[:2]

matrix = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
rotated_img = cv2.warpAffine(img, matrix, (w, h))

small_img = cv2.resize(img, (200, 200))

flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)

cv2.imshow('Original', img)
cv2.imshow('Rotated', rotated_img)
cv2.imshow('Resized', small_img)
cv2.imshow('Flip H', flip_h)
cv2.imshow('Flip V', flip_v)
cv2.waitKey(0)
cv2.destroyAllWindows()


#QUESTION 2.1
import cv2

img = cv2.imread("C:\Users\DHARSHANA D\OneDrive\Pictures\Screenshots\Screenshot 2025-12-20 011807.png", 0)

ret1, t1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
t2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
ret3, t3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Binary', t1)
cv2.imshow('Adaptive', t2)
cv2.imshow('Otsu', t3)
cv2.waitKey(0)
cv2.destroyAllWindows()


#QUESTION 2.2
import cv2

img = cv2.imread("C:\Users\DHARSHANA D\OneDrive\Pictures\Screenshots\Screenshot 2025-12-20 011807.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

print("Number of contours found:")
print(len(contours))

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
