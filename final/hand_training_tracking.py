# Import Libraries
import dlib
import glob
import cv2
import os
import sys
import  time
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil
from imutils import face_utils
import math
from OBJFileLoader import *
# If cleanup is True then the new images and annotations will be appended to previous ones
# If False then all previous images and annotations will be deleted.
cleanup = True
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
detector_face = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
# Set the window to a normal one so we can adjust it
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Resize the window and adjust it to the center
# This is done so we're ready for capturing the images.
cv2.resizeWindow('frame', 1920,1080)
cv2.moveWindow("frame", 0,0)

# Initialize webcam
cap = cv2.VideoCapture(0)
window_name = "tracking"
# Initalize sliding window's x1,y1
x1 ,y1 = 0,0

# These will be the width and height of the sliding window.
window_width = 190#140  
window_height = 190

# We will save images after every 4 frames
# This is done so we don't have lot's of duplicate images
skip_frames = 3
frame_gap = 0

# This is the directory where our images will be stored
# Make sure to change both names if you're saving a different Detector
directory = 'train_images_h'
box_file = 'boxes_h.txt'


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

# If cleanup is True then delete all imaages and bounding_box annotations.
if cleanup:
    
    # Delete the images directory if it exists
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    # Clear up all previous bounding boxes
    open(box_file, 'w').close()
    
    # Initialize the counter to 0
    counter = 0
    
elif os.path.exists(box_file):

    # If cleanup is false then we must append the new boxes with the old
    with open(box_file,'r') as text_file:
        box_content = text_file.read()
        
    # Set the counter to the previous highest checkpoint
    counter = int(box_content.split(':')[-2].split(',')[-1])

# Open up this text file or create it if it does not exists
fr = open(box_file, 'a')

# Create our image directory if it does not exists.
if not os.path.exists(directory):
   os.mkdir(directory)

# Initial wait before you start recording each row
initial_wait = 0
        
# Start the loop for the sliding window
while(True):
    
    # Start reading from camera
    ret, frame = cap.read()
    if not ret:
        print("failed")
        break
        
    # Invert the image laterally to get the mirror reflection.
    frame = cv2.flip( frame, 1 )
    
    # Make a copy of the original frame
    orig = frame.copy()    
    
    # Wait the first 50 frames so that you can place your hand correctly
    if initial_wait > 60:
        
        # Increment frame_gap by 1.
        frame_gap +=1  
    
        # Move the window to the right by some amount in each iteration.    
        if x1 + window_width < frame.shape[1]:
            x1 += 4
            time.sleep(0.1)            
            
        elif y1 + window_height + 270 < frame.shape[0]:

            # If the sliding_window has reached the end of the row then move down by some amount.
            # Also start the window from start of the row
            y1 += 200    
            x1 = 0

            # Setting frame_gap and init_wait to 0.
            # This is done so that the user has the time to place the hand correctly
            # in the next row before image is saved.
            frame_gap = 0
            initial_wait = 0
            
        # Break the loop if we have gone over the whole screen.
        else:
            break
              
    else: 
        initial_wait += 1

    # Save the image every nth frame.
    if frame_gap == skip_frames:

        # Set the image name equal to the counter value
        img_name = str(counter)  + '.png'

        # Save the Image in the defined directory
        img_full_name = directory + '/' + str(counter) +  '.png'
        cv2.imwrite(img_full_name, orig)
        
        # Save the bounding box coordinates in the text file.
        fr.write('{}:({},{},{},{}),'.format(counter, x1, y1, x1+window_width, y1+window_height))

        # Increment the counter 
        counter += 1

        # Set the frame_gap back to 0.
        frame_gap = 0

    # Draw the sliding window
    cv2.rectangle(frame,(x1,y1),(x1+window_width,y1+window_height),(0,255,0),3)
    
    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
                break

# Release camera and close the file and window
cap.release()
cv2.destroyAllWindows()
fr.close()

# In this dictionary our images and annotations will be stored.
data = {}

# Get the indexes of all images.
image_indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(directory)]

# Shuffle the indexes to have random train/test split later on.
np.random.shuffle(image_indexes)

# Open and read the content of the boxes.txt file
f = open(box_file, "r")
box_content = f.read()

# Convert the bounding boxes to dictionary in the format `index: (x1,y1,x2,y2)` ...
box_dict =  eval( '{' +box_content + '}' )

# Close the file
f.close()

# Loop over all indexes
for index in image_indexes:
    
    # Read the image in memmory and append it to the list
    img = cv2.imread(os.path.join(directory, str(index) + '.png'))    
    
    # Read the associated bounding_box
    bounding_box = box_dict[index]
    
    # Convert the bounding box to dlib format
    x1, y1, x2, y2  = bounding_box
    dlib_box = [ dlib.rectangle(left=x1 , top=y1, right=x2, bottom=y2) ]
    
    # Store the image and the box together
    data[index] = (img, dlib_box)


# This is the percentage of data we will use to train
# The rest will be used for testing
percent = 0.8

# How many examples make 80%.
split = int(len(data) * percent)

# Seperate the images and bounding boxes in different lists.
images = [tuple_value[0] for tuple_value in data.values()]
bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

# Initialize object detector Options
options = dlib.simple_object_detector_training_options()

# I'm disabling the horizontal flipping, becauase it confuses the detector if you're training on few examples
# By doing this the detector will only detect left or right hand (whichever you trained on). 
options.add_left_right_image_flips = False

# Set the c parameter of SVM equal to 5
# A bigger C encourages the model to better fit the training data, it can lead to overfitting.
# So set an optimal C value via trail and error.
options.C = 5

# Note the start time before training.
st = time.time()

# You can start the training now
detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)

# Print the Total time taken to train the detector
print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

file_name = 'Head_Detector.svm'
detector.save(file_name)



detector = dlib.train_simple_object_detector(images, bounding_boxes, options)
detector.save(file_name)

#file_name = 'Hand_Detector.svm'

# Load our trained detector 
detector = dlib.simple_object_detector(file_name)

# Set the window name
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setting the downscaling size, for faster detection
# If you're not getting any detections then you can set this to 1
scale_factor = 2.0

# Initially the size of the hand and its center x point will be 0
size, center_x = 0,0

# Initialize these variables for calculating FPS
fps = 0 
frame_counter = 0
start_time = time.time()

# Set the while loop
while(True):
    
    # Read frame by frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Laterally flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Calculate the Average FPS
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    
    # Create a clean copy of the frame
    copy = frame.copy()  
    
    # Downsize the frame.
    new_width = int(frame.shape[1]/scale_factor)
    new_height = int(frame.shape[0]/scale_factor)
    resized_frame = cv2.resize(copy, (new_width, new_height))
    
    # Detect with detector
    detections = detector(resized_frame)
    
    # Loop for each detection.
    for detection in (detections):    
        
        # Since we downscaled the image we will need to resacle the coordinates according to the original image.
        x1 = int(detection.left() * scale_factor )
        y1 =  int(detection.top() * scale_factor )
        x2 =  int(detection.right() * scale_factor )
        y2 =  int(detection.bottom()* scale_factor )
        
        # Draw the bounding box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2 )
        cv2.putText(frame, 'Hand Detected', (x1, y2+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get faces into webcam's image
        rects = detector_face(gray, 0)
        
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

        # Calculate size of the hand. 
        size = int( (x2 - x1) * (y2-y1) )
        
        # Extract the center of the hand on x-axis.
        center_x = x2 - x1 // 2
    
    # Display FPS and size of hand
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)

    # This information is useful for when you'll be building hand gesture applications
    cv2.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(frame, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    
    # Display the image
    cv2.imshow('frame',frame)
    cv2.imshow(window_name, frame)              
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Relase the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
