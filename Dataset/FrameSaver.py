import cv2

vidcap = cv2.VideoCapture('Data/DJI_0199.MOV')

frame_count = 0
number_frame = 0
image_amount = 0

while vidcap.isOpened():
    ret, frame = vidcap.read()
    if frame_count == 0:
        cv2.imwrite("Dataset/First.jpg", frame)
    frame_count = frame_count + 1
    if (frame_count > 1200):
        #alt_frame_count = frame_count - 1200
        #if (alt_frame_count % 25) == 0:
        
        if (frame_count % 25) == 0:
            cv2.imwrite("Dataset/Frames/%d.jpg" % frame_count, frame)

        #elif (alt_frame_count == 1):
            #cv2.imwrite("Dataset/Second.jpg", frame)


