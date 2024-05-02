import cv2
import numpy as np

img1 = cv2.imread("Dataset/Frames/0.jpg")
img2 = cv2.imread("Dataset/Frames/1.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_img1, des_img1 = sift.detectAndCompute(gray1, None)
kp_img2, des_img2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(des_img1, des_img2)

print("Number of matching features: ", len(matches))
        
img3 = cv2.drawMatches(img1, kp_img1, img2, kp_img2,matches, None, 
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("Map initialization/Matching_Features.png", img3)

Images = np.concatenate((img1, img2), axis=1)
cv2.imwrite("Map initialization/Side_By_Side.png", Images)