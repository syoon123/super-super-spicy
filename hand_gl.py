# Import Libraries
import dlib
import cv2
import sys
import  time
import numpy as np
from imutils import face_utils
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from fasterobj import OBJ

INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [ 1.0, 1.0, 1.0, 1.0]])
NEAR = 0.1
FAR = 100.0

# Window dimensions
WIDTH = 1280
HEIGHT = 720

# Sunglasses file
SUNGLASSES = 'Sunglasses.obj'

# Draw face detection rectangles
FACE_RECTS = True

# Hand Detection
HAND_INTEGRATION = True
HAND_DETECTOR = 'Head_detector.svm'


class FromVideo:
    def __init__(self):
        # Initialize webcam and start thread
        self.cap = cv2.VideoCapture(0)
        self.selected = False

        # Load trained hand detector
        self.detector_hand = dlib.simple_object_detector(HAND_DETECTOR)
        self.scale_factor = 2.0
        self.size, self.center_x = 0, 0
        self.fps = 0
        self.frame_counter = 0
        self.start_time = time.time()
        self.started = False

        # Initialize shapes
        self.sunglasses = OBJ(SUNGLASSES)
        self.texture_background = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                                        [-1.0,-1.0,-1.0,-1.0],
                                        [-1.0,-1.0,-1.0,-1.0],
                                        [ 1.0, 1.0, 1.0, 1.0]])

        # Initialize trackers
        self.tracker = dlib.correlation_tracker() # for face
        self.trackers = []
        for i in range(6):
            self.trackers.append(dlib.correlation_tracker()) # trackers for key landmarks on face for pose estimation

        self.width = None
        self.height = None

        self.prev_r = None
        self.prev_t = None

        self.count = 1

        self.x_axis = 0.0
        self.z_axis = 0.0
        self.texture_cube = None

        p = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)

    def keyboard(self, key, x, y):
        if key.decode() == 'q':
            sys.exit()

    def init_gl(self, Width, Height):
        self.width = Width
        self.height = Height
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width)/float(Height), NEAR, FAR)
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        glDisable(GL_TEXTURE_2D)

    def handle_background(self, image):
        # Convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = cv2.flip(bg_image, 1)
        bg_image = Image.fromarray(bg_image)
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # Create background texture
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-self.width * 0.003, self.height * 0.003, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(self.width * 0.003, self.height * 0.003, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(self.width * 0.003, -self.height * 0.003, 0.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-self.width * 0.003, -self.height * 0.003, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def render(self):
        _, image = self.cap.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray, 1)

        rects = self.detector(gray, 0)
        pos_available = False

        indices = [30, 8, 36, 45, 48, 54]

        if len(rects) > 0:
            pos_available = True
            self.tracker.start_track(image, rects[0])
            shape = self.predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            x, y = (shape[36] + shape[45]) / 2
            shapes = np.array([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])

            p = self.tracker.get_position()
            if FACE_RECTS:
                cv2.rectangle(image, (self.width - int(p.left()), int(p.top())),
                              (self.width - int(p.right()), int(p.bottom())), (255, 0, 0), 2)
            for i in range(len(indices)):
                self.selected = True
                cx = shape[indices[i]][0]
                cy = shape[indices[i]][1]
                self.trackers[i].start_track(image, dlib.rectangle(cx - 20, cy - 20, cx + 20, cy + 20))
            for i in range(len(indices)):
                p = self.trackers[i].get_position()
                if FACE_RECTS:
                    cv2.rectangle(image, (self.width - int(p.left()), int(p.top())),
                                  (self.width - int(p.right()), int(p.bottom())), (255, 0, 0), 2)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glPushMatrix()
            glTranslatef(0.0, 0.0, -5)
            self.handle_background(image)
            glPopMatrix()
        else:
            if self.selected:
                self.count += 1
                pos_available = True
                shapes = []
                self.tracker.update(image)
                p = self.tracker.get_position()
                cv2.rectangle(image, (self.width - int(p.left()), int(p.top())), (self.width - int(p.right()), int(p.bottom())), (0, 255, 0), 2)
                for i in range(len(indices)):
                    self.trackers[i].update(image)
                    p = self.trackers[i].get_position()
                    if FACE_RECTS:
                        cv2.rectangle(image, (self.width - int(p.left()), int(p.top())), (self.width - int(p.right()), int(p.bottom())), (0, 255, 0), 2)
                    shapes.append(((p.left() + p.right()) / 2., (p.top() + p.bottom()) / 2.))

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
                glPushMatrix()
                glTranslatef(0.0, 0.0, -5)
                self.handle_background(image)
                glPopMatrix()

        # Laterally flip the frame
        frame = cv2.flip(image, 1)

        # Calculate the Average FPS
        self.frame_counter += 1
        self.fps = (self.frame_counter / (time.time() - self.start_time))

        # Create a clean copy of the frame
        copy = frame.copy()

        # Downsize the frame.
        new_width = int(frame.shape[1]/self.scale_factor)
        new_height = int(frame.shape[0]/self.scale_factor)
        resized_frame = cv2.resize(copy, (new_width, new_height))

        # Detect with detector
        detections = self.detector_hand(resized_frame)

        # Loop for each detection.
        for detection in (detections):

            # Since we downscaled the image we will need to resacle the coordinates according to the original image.
            x1 = int(detection.left() * self.scale_factor )
            y1 = int(detection.top() * self.scale_factor )
            x2 = int(detection.right() * self.scale_factor )
            y2 = int(detection.bottom()* self.scale_factor )

            # Draw the bounding box
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2 )
            # cv2.putText(frame, 'Hand Detected', (x1, y2+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)

            if not self.started:
                self.started = True
                pos_available = True
            # Calculate size of the hand.
            self.size = int( (x2 - x1) * (y2-y1) )

            # Extract the center of the hand on x-axis.
            self.center_x = x2 - x1 // 2
        if len(detections) == 0:
            pos_available = False

        if pos_available or not HAND_INTEGRATION:
            r, t, c = self.estimate_pose(image, shapes)
            if self.prev_r is None or self.prev_t is None:
                self.prev_r = r
                self.prev_t = t
            else:
                if (np.dot(r, self.prev_r) / (np.linalg.norm(r) * np.linalg.norm(self.prev_r)) < .2):
                    shapes[4] = (shapes[4][0], (shapes[4][1] + shapes[0][1]) / 2.)
                    shapes[5] = (shapes[5][0], (shapes[5][1] + shapes[0][1]) / 2.)
                    r, t, c = self.estimate_pose(image, shapes)
                self.prev_r = r
                self.prev_t = t

            # COMMENTED OUT BECAUSE IT DOESN'T WORK :(
            # alpha = c[0][0]
            # beta = c[1][1]
            # x_0 = c[0][2]
            # y_0 = c[1][2]
            # persp = np.array([
            #     [alpha, 0, 0, -x_0],
            #     [0, beta, 0, -y_0],
            #     [0, 0, NEAR + FAR, NEAR * FAR],
            #     [0, 0, 1, 0]
            # ])
            # glMatrixMode(GL_PROJECTION)
            # glPopMatrix()
            # glLoadIdentity()
            # glOrtho(0, self.width, self.height, 0, NEAR, FAR)
            # glMultMatrixf(persp)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            z = t[2] * -750 / (self.width * self.height)
            glTranslatef(0, 0, z)
            rmtx = cv2.Rodrigues(r)[0]
            view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], 1.75 * t[0] / self.width],
                                    [rmtx[1][0], rmtx[1][1], rmtx[1][2], 1.75 * t[1] / self.height],
                                    [rmtx[2][0], rmtx[2][1], rmtx[2][2], 0],
                                    [0.0, 0.0, 0.0, 1.0]])
            view_matrix = view_matrix * INVERSE_MATRIX
            view_matrix = np.transpose(view_matrix)
            glMultMatrixf(view_matrix)
            glRotate(90, 1, 0, 0)
            glRotate(180, 0, 1, 0)
            z_t = 1.1
            glScalef(z_t * 0.145, z_t * 0.145, z_t * 0.145)
            glTranslatef(0.0, -2.0 * z_t, 0.0)
            glEnable(GL_LIGHTING)
            glMaterialfv(GL_FRONT, GL_SPECULAR, [1, 1, 1, 0.35])
            glLightfv(GL_LIGHT0, GL_POSITION, (0.15, 0.3, 0.8, 0.0))
            glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT1, GL_POSITION, (-0.15, 0.3, 0.8, 0.0))
            glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
            glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
            glEnable(GL_LIGHT1)

            self.sunglasses.render()
            glPopMatrix()
            glDisable(GL_LIGHTING)

        glEnable(GL_DEPTH_TEST)
        glutSwapBuffers()

    def estimate_pose(self, im, shapes):
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

        return rotation_vector, translation_vector, camera_matrix

    def run(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(WIDTH, HEIGHT)
        glutInitWindowPosition(100, 100)
        glutCreateWindow("OpenGL")
        glutDisplayFunc(self.render)
        glutIdleFunc(self.render)
        glutKeyboardFunc(self.keyboard)

        self.init_gl(WIDTH, HEIGHT)
        glutMainLoop()


a = FromVideo()
a.run()
