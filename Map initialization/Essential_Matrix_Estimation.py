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

points1_temp = []
points2_temp = []
match_indices_temp = []
for idx, m in enumerate(matches):
    points1_temp.append(kp_img1[m.queryIdx].pt)
    points2_temp.append(kp_img2[m.trainIdx].pt)
    match_indices_temp.append(idx)

points1 = np.float32(points1_temp)
points2 = np.float32(points2_temp)
match_indices = np.int32(match_indices_temp)
ransacReprojecThreshold = 1
confidence = 0.99

Camera_matrix = np.array([[2676, 0., 3840 / 2 - 35.24], 
            [0.000000000000e+00, 2676., 2160 / 2 - 279],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

essentialMatrix, mask = cv2.findEssentialMat(
    points1= points1, 
    points2= points2, 
    cameraMatrix= Camera_matrix,
    method= cv2.FM_RANSAC, 
    threshold=ransacReprojecThreshold, 
    prob= confidence)

match_indices = match_indices[mask.ravel()==1]
filtered_matches = []
for index in match_indices:
    m = matches[index]
    filtered_matches.append(m)


img3 = cv2.drawMatches(img1, kp_img1,
        img2, kp_img2,
        filtered_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("Map initialization/Matching_Features_Filtered.png", img3)

print("Mathcing features filtered: ", len(filtered_matches))

print(essentialMatrix)

# Calculating distances
distances = []
for m, pt1, pt2 in zip(filtered_matches, points1, points2):
    pt1_homogeneous = np.array([pt1[0], pt1[1], 1])
    pt2_homogeneous = np.array([pt2[0], pt2[1], 1])
    epipolar_line = np.dot(essentialMatrix, pt1_homogeneous)
    distance = np.abs(np.dot(epipolar_line, pt2_homogeneous)) / np.linalg.norm(epipolar_line[:2])
    distances.append(distance)

# Summary statistics
mean_distance = np.mean(distances)
std_distance = np.std(distances)

print("Mean distance:", mean_distance)
print("Standard deviation of distances:", std_distance)