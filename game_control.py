# Grabbing our dependencies
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from keyboard_control import PressKey, ReleaseKey
from keyboard_control import W, A, S, D
# define the lower and upper boundaries of the "green" in the HSV color space
color_Lower = (29, 86, 6)
color_Upper = (64, 255, 255)
#Creating a VideoCapture object to read video from the primary camera
vs = VideoStream(src=0).start()


initial = True
flag = False
current_key_pressed = set()
circle_radius = 30
windowSize = 160
lr_counter = 0
#allow the system to sleep for 3 sec before webcam starts
time.sleep(3)
# Continue till user wishes to quit
while True:
    keyPressed = False
    keyPressed_lr = False
    # Get the current frame
    frame = vs.read() 
    # Get the dimensions of the current frame
    height, width = frame.shape[:2]
    # Downsize the frame to a width of 600 px for faster processing leading to a increase in FPS
    frame = imutils.resize(frame, width=600) 
    # Blur the frame to reduce any high frequency noise and increase focus on structural objects
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # Converting the color space from BGR to HSV as BGR is more sensitive to light
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # crteate a binary mask for the given color 
    mask = cv2.inRange(hsv, color_Lower, color_Upper)
    # Further Blobs are removed by erosion and dilation
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Divide the frame into two halves so that we can have one half control the acceleration/deceleration and other half control the left/right steering.
    left_mask = mask[:, 0:width//2, ]
    right_mask = mask[:, width//2:, ]
    # Find the contours on the left of the frame
    cnts_left = cv2.findContours(left_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_left = imutils.grab_contours(cnts_left)
    center_left = None
    # Find the contours on the right of the frame
    cnts_right = cv2.findContours(right_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_right = imutils.grab_contours(cnts_right)
    center_right = None
    key_count = 0
    key_pressed = 0
    # If there is atleast one contour on the left half of the mask
    if len(cnts_left) > 0:
        # Calculate the largest contour
        c1 = max(cnts_right, key=cv2.contourArea)
        # Calculate the minimum enclosing circle of the blob
        ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
        # compute the centroids
        M = cv2.moments(c1)
        # find the center from the moments 0.000001 is added to the denominator so that divide by
        # zero exception doesn't occur
        center_left = (int(M["m10"] / (M["m00"]+0.000001)),int(M["m01"] / (M["m00"]+0.000001)))
        # Check if the radius of the minimum enclosing circle is sufficiently large
        if radius1 > circle_radius:
            # Draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius1),(0, 0, 255), 2)
            cv2.circle(frame, center_left, 5, (0, 0, 255), -1)
            #Bottom Left region
            if center_left[1] > 250:
                cv2.putText(frame, 'Break Applied', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                PressKey(S)
                break_pressed = True
                current_key_pressed.add(S)
                key_pressed = S
                keyPressed = True
                key_count = key_count+1

    # If there is atleast one contour on the right half of the mask
    if len(cnts_right) > 0:
        # Calculate the largest contour
        c2 = max(cnts_right, key=cv2.contourArea) 
        # Calculate the minimum enclosing circle of the blob
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        # compute the centroids
        M2 = cv2.moments(c2)
        center_right = (int(M2["m10"] / (M2["m00"]+0.000001)),int(M2["m01"] / (M2["m00"]+0.000001)))
        center_right = (center_right[0]+width//2, center_right[1])
        # Check if the radius of the minimum enclosing circle is sufficiently large
        if radius2 > circle_radius:
            cv2.circle(frame, (int(x2)+width//2, int(y2)), int(radius2),(0, 255, 0), 2)
            cv2.circle(frame, center_right, 5, (0, 255, 0), -1)
            #Bottom Right region
            if center_right[1] > 250:
                cv2.putText(frame, 'Acc. Applied', (350, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                PressKey(W)
                key_pressed = W
                accelerator_pressed = True
                keyPressed = True
                current_key_pressed.add(W)
                key_count = key_count+1

    frame_copy = frame.copy()
    # Put text on the bottom left region rectangle
    frame_copy = cv2.rectangle(frame_copy, (0, height//2), (width//2, width), (255, 255, 255), 1)
    cv2.putText(frame_copy, 'Break', (10, 280),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # Put text on the bottom right region rectangle
    frame_copy = cv2.rectangle(frame_copy, (width//2, height//2), (width, height), (255, 255, 255), 1)
    cv2.putText(frame_copy, 'Acceleration', (330, 280),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # Display our frame
    cv2.imshow("Frame", frame_copy)
    # If no key is pressed
    if not keyPressed and len(current_key_pressed) != 0:
        for key in current_key_pressed:
            # Release the pressed key
            ReleaseKey(key) 
        current_key_pressed = set()
    # If a key is pressed
    elif key_count == 1 and len(current_key_pressed) == 2:
            for key in current_key_pressed:
                if key_pressed != key:
                    ReleaseKey(key) # Release previous frame key
            current_key_pressed = set()
            for key in current_key_pressed:
                ReleaseKey(key)
            current_key_pressed = set()
    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stop()
# close all windows
cv2.destroyAllWindows()
