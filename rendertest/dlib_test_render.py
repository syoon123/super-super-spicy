#!/usr/bin/env python
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math
# import pywavefront
from OBJFileLoader import *

import os
os.chdir("C:\\Users\\sux3\\Desktop\\dlib_test")

# glasses_obj = pywavefront.Wavefront('Sunglasses.obj')
glasses_obj = OBJ(filename = 'Sunglasses.obj')

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

win = dlib.image_window()

def projection_matrix(camera_parameters, homography):
    """
     From the camera calibration matrix and the estimated homography
     compute the 3D projection matrix
     """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    
    return np.dot(camera_parameters, projection)
    
def render(img, obj, projection, rect, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    # h, w = model.shape
    h = rect.bottom() - rect.top()
    w = rect.right() - rect.left()

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
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

    # print('Camera Matrix :',camera_matrix)

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs  )#, flags=cv2.CV_ITERATIVE)

    # print('Rotation Vector:',rotation_vector)
    # print('Translation Vector:',translation_vector)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    #vector starts at P1 and ends at P2
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255,0,0), 2)
    
    #get homography
    R, _ = cv2.Rodrigues(rotation_vector) #get rotation matrix
    RT = np.vstack([R.T,translation_vector.T]).T #append translation vector
    H = np.dot(camera_matrix,RT)
    
    P = projection_matrix(camera_matrix, H)
    
    
    im_render = render(im, glasses_obj, P, rect)
    # print(H.shape)#3x2
    return im
    # cv2.waitKey(0)


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
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
        # print(rect) # [(251,194) (509, 452)]
        
        
        #pose estimation
        image = estimate_pose(image,shape, rect)
    
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