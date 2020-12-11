import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from imutils import face_utils
import dlib
import numpy as np
import os
import sys
# from OBJFileLoader import OBJ
from fasterobj import OBJ, OBJ_array, OBJ_vbo

global sunglasses
sunglasses = OBJ(filename='Sunglasses.obj')


# Window dimensions
width = 1280
height = 720
nRange = 1.0

global capture
capture = None
global texture_background
texture_background = None
global rotation_vector
rotation_vector = None
global translation_vector
translation_vector = None
global camera_matrix
camera_matrix = None
global dist_coeffs
dist_coeffs = None

INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [ 1.0, 1.0, 1.0, 1.0]])

tracker = dlib.correlation_tracker()

tracker1 = dlib.correlation_tracker()
tracker2 = dlib.correlation_tracker()
tracker3 = dlib.correlation_tracker()
tracker4 = dlib.correlation_tracker()
tracker5 = dlib.correlation_tracker()
tracker6 = dlib.correlation_tracker()
trackers = [tracker1, tracker2, tracker3, tracker4, tracker5, tracker6]

# load facial landmarks
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)  # enable the lighting
    glEnable(GL_LIGHT0)  # enable LIGHT0, our Diffuse Light
    glShadeModel(GL_SMOOTH)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)

    glEnable(GL_TEXTURE_2D)
    global texture_background
    texture_background = glGenTextures(1)


def idle():
    global capture
    _, image = capture.read()

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)

    if len(rects) > 0:
        r = rects[0]
        # tracker.start_track(image, r)
        shape = predictor(gray, r)
        shape = face_utils.shape_to_np(shape)
        indices = [30, 8, 36, 45, 48, 54]
        shapes = np.array([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])

        # for i in range(len(indices)):
        #     cx = shape[indices[i]][0]
        #     cy = shape[indices[i]][1]
        #     trackers[i].start_track(image, dlib.rectangle(cx - 20, cy - 20, cx + 20, cy + 20))
        #
        r, t, k, c = estimate_pose(image, shapes)
        global rotation_vector
        rotation_vector = r
        global translation_vector
        translation_vector = t
        global camera_matrix
        camera_matrix = k
        global dist_coeffs
        dist_coeffs = c

    image = cv2.flip(image, 0)
    image = cv2.flip(image, 1)

    global texture_background
    glBindTexture(GL_TEXTURE_2D, texture_background)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_BGR,
        GL_UNSIGNED_BYTE,
        image
    )

    glutPostRedisplay()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)

    # Draw background
    global texture_background
    glBindTexture(GL_TEXTURE_2D, texture_background)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(width, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(width, height)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, height)
    glEnd()

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Build view matrix
    global rotation_vector, translation_vector, camera_matrix
    # if rotation_vector is not None:
    #     print('Rotation vector: ' + str(rotation_vector))
    # if translation_vector is not None:
    #     print('Translation vector: ' + str(translation_vector))
    # if camera_matrix is not None:
    #     print('Camera matrix: ' + str(camera_matrix))
    if not(rotation_vector is None or
           translation_vector is None or
           camera_matrix is None):
        rmtx = cv2.Rodrigues(rotation_vector)[0]
        view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], translation_vector[0]],
                                [rmtx[1][0], rmtx[1][1], rmtx[1][2], translation_vector[1]],
                                [rmtx[2][0], rmtx[2][1], rmtx[2][2], translation_vector[2]],
                                [0.0, 0.0, 0.0, 1.0]])
        view_matrix = view_matrix * INVERSE_MATRIX
        view_matrix = np.transpose(view_matrix)

        glPushMatrix()
        glLoadMatrixd(view_matrix)

        global sunglasses
        sunglasses.basic_render()

        glPopMatrix()

    glFlush()
    glutSwapBuffers()


def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()
    # allows for reshaping the window without distorting shape

    if w <= h:
        glOrtho(-nRange, nRange, -nRange * h / w, nRange * h / w, -nRange, nRange)
    else:
        glOrtho(-nRange * w / h, nRange * w / h, -nRange, nRange, -nRange, nRange)

    glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()


def keyboard(key, x, y):
    if key.decode() == 'q':
        sys.exit()


def estimate_pose(im, shapes):
    size = im.shape

    image_points = np.array([
        shapes[0],     # Nose tip
        shapes[1],     # Chin
        shapes[2],     # Left eye left corner
        shapes[3],     # Right eye right corner
        shapes[4],     # Left Mouth corner
        shapes[5],      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points,
                                                                  image_points,
                                                                  camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_vector = np.squeeze(rotation_vector)
    translation_vector = np.squeeze(translation_vector)

    return rotation_vector, translation_vector, camera_matrix, dist_coeffs


def main():
    global capture
    capture = cv2.VideoCapture(0)

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("OpenGL + OpenCV")
    glutKeyboardFunc(keyboard)

    init()
    glutMainLoop()


main()


