#!/usr/bin/env python
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math

from OBJFileLoader import *

import os
# os.chdir(r"./super-super-spicy/rendertest")

#load obj objects
obj1 = OBJ(filename = 'Sunglasses.obj')
obj2 = OBJ(filename = 'Hat.obj')

#load facial landmarks
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

eye_detector = dlib.get_frontal_face_detector()
predictor2 = dlib.shape_predictor("eye_predictor.dat")

hand_detector = dlib.simple_object_detector("detector2.svm")

#start video capture
cap = cv2.VideoCapture(0)
win = dlib.image_window()
    
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
    im_render = render(im, obj1, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,0.6,2.5], scale = 140, color = (0, 0, 0))
    im_render = render(im_render, obj2, rect, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point_offset = [0,-1,4], scale = 700, color = (0, 0, 0))
    
    # return im
    return im_render


while True:

    # Getting out image by webcam 
    _, image = cap.read()
    
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

    rects_eye = eye_detector(gray, 0)
    for r in rects_eye:
        (x, y, w, h) = face_utils.rect_to_bb(r)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        shape = predictor2(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (sX, sY) in shape:
            cv2.circle(image, (sX, sY), 1, (0, 0, 255), -1)

    rects_hand = hand_detector(gray, 1)
    print(rects_hand)
    for (i, r) in enumerate(rects_hand):
        (x, y, w, h) = face_utils.rect_to_bb(r)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # win.clear_overlay()    
    # win.set_image(image)
    # win.add_overlay(rects)
    
    # Show the image
    cv2.imshow("Output", image)
    
    # cv2.imwrite('testface.jpg',image)    
    
    #exit with esc keypress
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
