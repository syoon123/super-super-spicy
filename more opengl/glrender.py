from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from imutils import face_utils
import cv2
from PIL import Image
import numpy as np

from imutils import face_utils
import dlib

from fasterobj import OBJ

# from OBJFileLoader import *

INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [ 1.0, 1.0, 1.0, 1.0]])

# Window dimensions
WIDTH = 1280
HEIGHT = 720

# Sunglasses file
SUNGLASSES = 'Sunglasses.obj'


class FromVideo:
    def __init__(self):
        # initialise webcam and start thread
        self.cap = cv2.VideoCapture(0)

        # initialise shapes
        self.sunglasses = OBJ(SUNGLASSES)
        self.texture_background = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                                        [-1.0,-1.0,-1.0,-1.0],
                                        [-1.0,-1.0,-1.0,-1.0],
                                        [ 1.0, 1.0, 1.0, 1.0]])
        self.width = None
        self.height = None

        self.x_axis = 0.0
        self.z_axis = 0.0

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
        gluPerspective(45.0,float(Width)/float(Height),0.1,100.0)
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        glDisable(GL_TEXTURE_2D)

    def handle_background(self, image):
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = cv2.flip(bg_image, 1)
        bg_image = Image.fromarray(bg_image)
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
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

    def draw_stuff(self):
        _, image = self.cap.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray, 1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glPushMatrix()
        glTranslatef(0.0,0.0,-5)
        self.handle_background(image)
        glPopMatrix()

        rects = self.detector(gray, 0)
        if (len(rects) > 0):
            shape = self.predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            x, y = (shape[36] + shape[45])/2
            shapes = np.array([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])
            r, t = self.estimate_pose(image, shapes)
            z = t[2] * -750/(self.width * self.height)
            glPushMatrix()
            glTranslatef(0, 0, z)
            rmtx = cv2.Rodrigues(r)[0]
            view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], 1.75*t[0]/self.width],
                                    [rmtx[1][0], rmtx[1][1], rmtx[1][2], (1.75*t[1] + (y - shape[30][1]))/self.height],
                                    [rmtx[2][0], rmtx[2][1], rmtx[2][2], 0],
                                    [0.0, 0.0, 0.0, 1.0]])
            view_matrix = view_matrix * INVERSE_MATRIX
            view_matrix = np.transpose(view_matrix)
            glMultMatrixf(view_matrix)
            glRotate(90, 1, 0, 0)
            glRotate(180, 0, 1, 0)
            glScalef(0.145, 0.145, 0.145)
            glEnable(GL_LIGHTING)
            glMaterialfv(GL_FRONT, GL_SPECULAR, [0.35, 0.35, 0.35, 1.0])
            glLightfv(GL_LIGHT0, GL_POSITION, (-0.15, 0.3, 0.8, 0.0))
            glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT1, GL_POSITION, (0.15, 0.3, 0.8, 0.0))
            glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
            glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
            glEnable(GL_LIGHT1)
            self.sunglasses.render()
            glPopMatrix()
            glDisable(GL_LIGHT0)
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

        return rotation_vector, translation_vector

    def run(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(WIDTH, HEIGHT)
        glutInitWindowPosition(100, 100)
        glutCreateWindow("OpenGL")
        glutDisplayFunc(self.draw_stuff)
        glutIdleFunc(self.draw_stuff)
        glutKeyboardFunc(self.keyboard)

        self.init_gl(WIDTH, HEIGHT)
        glutMainLoop()


def main():
    a = FromVideo()
    a.run()


if __name__ == "__main__":
    main()
