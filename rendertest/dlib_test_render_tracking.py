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

tracker = dlib.correlation_tracker()

tracker1 = dlib.correlation_tracker()
tracker2 = dlib.correlation_tracker()
tracker3 = dlib.correlation_tracker()
tracker4 = dlib.correlation_tracker()
tracker5 = dlib.correlation_tracker()
tracker6 = dlib.correlation_tracker()
trackers = [tracker1, tracker2, tracker3, tracker4, tracker5, tracker6]

#load facial landmarks
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#eye_detector = dlib.get_frontal_face_detector()
#predictor2 = dlib.shape_predictor("eye_predictor.dat")
#
#hand_detector = dlib.simple_object_detector("detector2.svm")

window_name = "tracking"

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
    

def estimate_pose(im,shapes,rect):

    size = im.shape
    
    #2D image points. If you change the image, you need to change vector

    #corners of eyes: 37, 46
    #nose tip: 31
    #corners of mouth: 49, 55
    #chin: 9

    #                             shape[30],     # Nose tip
    #                             shape[8],     # Chin
    #                             shape[36],     # Left eye left corner
    #                             shape[45],     # Right eye right corner
    #                             shape[48],     # Left Mouth corner
    #                             shape[54],      # Right mouth corner
    
    image_points = np.array([
                                shapes[0],     # Nose tip
                                shapes[1],     # Chin
                                shapes[2],     # Left eye left corner
                                shapes[3],     # Right eye right corner
                                shapes[4],     # Left Mouth corner
                                shapes[5],      # Right mouth corner
                                
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
    
shapes = []
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
selected = False
while True:

    # Getting out image by webcam 
    _, frame = cap.read()
    
    # Converting the image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    # for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
    if len(rects) > 0:
        r = rects[0]
        tracker.start_track(frame, r)
        shape = predictor(gray, r)
        shape = face_utils.shape_to_np(shape)
        indices = [30, 8, 36, 45, 48, 54]
        shapes = np.array([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])

        for i in range(len(indices)):
            selected = True
            cx = shape[indices[i]][0]
            cy = shape[indices[i]][1]
            trackers[i].start_track(frame, dlib.rectangle(cx-20, cy-20, cx+20, cy+20))
            # kalmans[i].correct(np.array([np.float32(shape[indices[i]][0]), np.float32(shape[indices[i]][1])], np.float32))
        #pose estimation
        frame = estimate_pose(frame,shapes, r)
            #pose estimation
            # frame = estimate_pose(frame,shapes, rect)
            

    else: 
        if selected:
            # img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # img_bproject = cv2.calcBackProject(
            #         [img_hsv], [
            #             0, 1], crop_hist, [
            #             0, 180, 0, 255], 1)
            # # cv2.imshow(window_name2, img_bproject)
            # ret, track_window = cv2.CamShift(
            #         img_bproject, track_window, term_crit)

            # x, y, w, h = track_window
            # frame = cv2.rectangle(
            #     frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            indices = [30, 8, 36, 45, 48, 54]
            shapes = []
            tracker.update(frame)
            p = tracker.get_position()
#            rect = dlib.rectangle(int(p.left()), int(p.bottom()), int(p.right()), int(p.top()))
            cv2.rectangle(frame, (int(p.left()), int(p.top())), (int(p.right()), int(p.bottom())), (255, 0, 0), 2)
            for i in range(len(indices)):
            # for i in range(1):
                # img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # img_bproject = cv2.calcBackProject(
                #         [img_hsv], [
                #             0, 1], crop_hists[i], [
                #             0, 180, 0, 255], 1)
                
                # ret, track_windows[i] = cv2.CamShift(
                #     img_bproject, track_windows[i], term_crit)

                # x, y, w, h = track_windows[i]
                # frame = cv2.rectangle(
                #     frame, (x, y), (x + 5, y + 5), (255, 0, 0), 2)

                # pts = cv2.boxPoints(ret)
                # pts = np.int0(pts)
                # (cx, cy), radius = cv2.minEnclosingCircle(pts)

                # use to correct kalman filter
                # print(pts)

                # a = center(pts)
                # print(a)
                # kalman.correct(a)

                # get new kalman filter prediction

                # prediction = kalmans[i].predict()
                trackers[i].update(frame)
                p = trackers[i].get_position()
                print(p)
                cv2.rectangle(frame, (int(p.left()), int(p.top())), (int(p.right()), int(p.bottom())), (0, 255, 0), 2)

                # draw predicton on image - in GREEN

                # frame = cv2.rectangle(frame,
                #                     (int(prediction[0] - (0.5 * 5)),
                #                     int(prediction[1] - (0.5 * 5))),
                #                     (int(prediction[0] + (0.5 * 5)),
                #                     int(prediction[1] + (0.5 * 5))),
                #                     (0,
                #                         255,
                #                         0),
                #                     2)
                shapes.append(((p.left() + p.right()) / 2., (p.top() + p.bottom()) / 2.))
                # shapes.append((x, y))
            # shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))
            # shape = face_utils.shape_to_np(shape)
    
            
            #pose estimation
            frame = estimate_pose(frame,shapes, r)

    cv2.imshow(window_name, frame)
    
    #exit with esc keypress
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
