#!/usr/bin/env python
from OBJFileLoader import *
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import math

import os
os.chdir("C:\\Users\\sux3\\Desktop\\dlib_test")

#load obj objects
obj1 = OBJ(filename = 'Sunglasses.obj')
obj2 = OBJ(filename = 'Hat.obj')

#load facial landmarks
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


# Hold the background frame for background subtraction.
background = None
# Hold the hand's data so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
frames_elapsed = 0
FRAME_HEIGHT = 500
FRAME_WIDTH = 700
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.1#0.5
OBJ_THRESHOLD = 30#40

# Our region of interest will be the top right part of the frame.
region_top = 0
region_bottom = int(1.5 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH *0.6)
region_right = FRAME_WIDTH

frames_elapsed = 0

#start video capture
cap = cv2.VideoCapture(0)
# win = dlib.image_window()

class HandData:
    top = (0,0)
    bottom = (0,0)
    left = (0,0)
    right = (0,0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    fingers = 0
    gestureList = []
    
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        isInFrame = False
        isWaving = False
        
    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    # def check_for_waving(self, centerX):
        # self.prevCenterX = self.centerX
        # self.centerX = centerX
        
        # if abs(self.centerX - self.prevCenterX > 3):
            # self.isWaving = True
        # else:
            # self.isWaving = False
            


def write_on_image(frame):
    # Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
    # so we can print on the screen which gesture is happening (or if the camera is calibrating).

    text = "searching..."

    if hand == None or hand.isInFrame == False:
        text = "No hand detected"
    else:
        # if hand.isWaving:
            # text = "Waving"
        if hand.fingers == 0:
            text = "Fist"
        elif hand.fingers == 1:
            text = "One Finger"
        elif hand.fingers == 2:
            text = "Two Fingers"
        elif hand.fingers == 3:
            text = "Three Fingers"
        elif hand.fingers == 4:
            text = "Four Fingers"
        else:
            text = "Five Fingers"
            
        # print(hand.fingers)
        
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 1,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    # cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1,cv2.LINE_AA)

    # Highlight the region of interest using a drawn rectangle.
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)
    
    

def get_region(frame):
    # Separate the region of interest from the rest of the frame.
    region = frame[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent frame noise from being labeled as an edge.
    region = cv2.GaussianBlur(region, (5,5), 0)

    return region
    
def get_average(region):
    # We have to use the global keyword because we want to edit the global variable.
    global background
    # If we haven't captured the background yet, make the current region the background.
    if background is None:
        background = region.copy().astype("float")
        return
    # Otherwise, add this captured frame to the average of the backgrounds.
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(region):
    # Here we use differencing to separate the background from the object of interest.
    global hand
    # Find the absolute difference between the background and the current frame.
    diff = cv2.absdiff(background.astype(np.uint8), region)

    # Threshold that region with a strict 0 or 1 ruling so only the foreground remains.
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Get the contours of the region, which will return an outline of the hand.
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    # If we didn't get anything, there's no hand.
    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False
        return
    # Otherwise return a tuple of the filled hand (thresholded_region), along with the outline (segmented_region).
    else:
        if hand is not None:
            hand.isInFrame = True
        segmented_region = max(contours, key = cv2.contourArea)
        return (thresholded_region, segmented_region)

def get_hand_data(thresholded_image, segmented_image):
    global hand
    
    # Enclose the area around the extremities in a convex hull to connect all outcroppings.
    convexHull = cv2.convexHull(segmented_image)
    
    # Find the extremities for the convex hull and store them as points.
    top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    
    # Get the center of the palm, so we can check for waving and find the fingers.
    centerX = int((left[0] + right[0]) / 2)
    
    if hand == None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)
        
    # if frames_elapsed % 6 == 0:
        # hand.check_for_waving(centerX)
        
    # We count the number of fingers up every frame, but only change hand.fingers if
    # 10 frames have passed, to prevent erratic gesture counts.
    
    hand.gestureList.append(count_fingers(thresholded_image))
    
    if frames_elapsed % 10 == 0:
        hand.fingers = most_frequent(hand.gestureList)
        hand.gestureList.clear()

def count_fingers(thresholded_image):

    # Find the height at which we will draw the line to count fingers.
    line_height = int(hand.top[1] + (0.3 * (hand.bottom[1] - hand.top[1])))
    
    # Get the linear region of interest along where the fingers would be.
    line = np.zeros(thresholded_image.shape[:2], dtype=int)
        
    # Draw a line across this region of interest, where the fingers should be.
    cv2.line(line, (thresholded_image.shape[1], line_height), (0, line_height), (255,0,0), 8)
        
    # Do a bitwise AND to find where the line intersected the hand -- this is where the fingers are.
    line = cv2.bitwise_and(thresholded_image, thresholded_image, mask = line.astype(np.uint8))
    
    # Get the line's new contours. The contours are basically just little lines formed by gaps 
    # in the big line across the fingers, so each would be a finger unless it's very wide.
    contours, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.imshow('line',line.copy())
    
    fingers = 0
    
    # Count the fingers by making sure the contour lines are "finger-sized", i.e. not too wide.
    # This prevents a "rock" gesture from being mistaken for a finger.
    for curr in contours:
        width = len(curr)
        
        # print('width',width)
        
        if width < 3 * abs(hand.right[0] - hand.left[0]) / 4 and width > 5:
            fingers += 1
    
    return fingers

    
def most_frequent(input_list):
    dict = {}
    count = 0
    most_freq = 0
    
    for item in reversed(input_list):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count :
            count, most_freq = dict[item], item
    
    return most_freq



def render(img, obj, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,0.6,2.5], scale = 140, color=(0, 0, 0)):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    
    # height and width of rect enclosing face
    h = rect.bottom() - rect.top()
    w = rect.right() - rect.left()
    

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        # points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        points = np.array([[p[0] + point_offset[0]*w, p[1] + point_offset[1]*h, p[2] - point_offset[2]*w]  for p in points])
        
        # dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        (dst, jacobian) = cv2.projectPoints(points.reshape(-1, 1, 3), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        imgpts = np.int32(dst)
       
        cv2.fillConvexPoly(img, imgpts, color)

    return img

def estimate_pose(im,shape,rect):

    size = im.shape
    
    #2D image points. If you change the image, you need to change vector

    #corners of eyes: 37, 46
    #nose tip: 31
    #corners of mouth: 49, 55
    #chin: 9
    
    image_points = np.array([
                                shape[30],     # Nose tip
                                shape[8],     # Chin
                                shape[36],     # Left eye left corner
                                shape[45],     # Right eye right corner
                                shape[48],     # Left Mouth corner
                                shape[54],      # Right mouth corner
                                
                            ], dtype="double")
   
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])


    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # for p in image_points:
        # cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    #vector starts at P1 and ends at P2
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    #draw projection vector
    # cv2.line(im, p1, p2, (255,0,0), 2)
    
    #render obj
    # im_render = render(im, obj1, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,0.6,2.5], scale = 140, color = (0, 0, 0))
    # im_render = render(im_render, obj2, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,-1,4], scale = 700, color = (0, 0, 0))
    
    color = (0,0,0)
    if hasattr(hand, 'fingers'):
        # print(hand.fingers)
        if hand.fingers == 0:
            color = (0,0,0)
        elif hand.fingers == 1:
            color = (255,0,0)
        elif hand.fingers == 2:
            color = (0,255,0)
        elif hand.fingers == 3:
            color = (0,0,255)
        elif hand.fingers == 4:
            color = (255,0,127)
        elif hand.fingers == 5:
            color = (255,255,255)
        
    im_render = render(im, obj1, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,0.6,2.5], scale = 140, color= color)
    im_render = render(im_render, obj2, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,-1,4], scale = 700, color= color)
    
    # return im
    return im_render


while True:

    # Getting out image by webcam 
    ret, image = cap.read()
    
    # Store frame and resize to desired window size
    image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Mirror image for ease of use
    image = cv2.flip(image, 1)
    
    # Separate the region of interest and prep it for edge detection.
    region = get_region(image)
    
    if frames_elapsed < CALIBRATION_TIME:
        get_average(region)
    else:
        region_pair = segment(region)
        if region_pair is not None:
            # If we have the regions segmented successfully, show them in another window for the user.
            (thresholded_region, segmented_region) = region_pair
            cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
            cv2.imshow("Segmented Image", region)
            
            get_hand_data(thresholded_region, segmented_region)

            
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
        # for (x, y) in shape:
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
        #pose estimation
        image = estimate_pose(image,shape, rect)
    
    # win.clear_overlay()    
    # win.set_image(image)
    # win.add_overlay(rects)


    write_on_image(image)
    
    # Show the image
    cv2.imshow("Output", image)
    
    frames_elapsed += 1
    
    # cv2.imwrite('testface.jpg',image)    
    
    #exit with esc keypress
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()




