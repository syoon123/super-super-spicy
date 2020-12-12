from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from imutils import face_utils
import cv2
from PIL import Image
import numpy as np

from imutils import face_utils
import dlib

# from fasterobj import OBJ, OBJ_array, OBJ_vbo

# from OBJFileLoader import *

class FromVideo:
    def __init__(self):
        # initialise webcam and start thread
        self.cap = cv2.VideoCapture(0)

        # initialise shapes
        # self.glasses = OBJ('Sunglasses.obj')
        # self.hat = OBJ('Hat.obj')
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
        self.texture_cube = None

        p = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)

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
        self.texture_cube = glGenTextures(1)

        image = Image.open("img2.png")
        ix = image.size[0]
        iy = image.size[1]
        image = image.tobytes("raw", "RGBA", 0, -1)
        glBindTexture(GL_TEXTURE_2D, self.texture_cube)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    def handle_background(self, image):
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)


    def draw_background(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd( )

    def draw_stuff(self):
        _, image = self.cap.read()
        self.handle_background(image)

        x, y = 0, 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            x, y = shape[45]

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-11.2)
        self.draw_background()
        glPopMatrix()

        # glColor4f(1.0, 1.0, 1.0, 1.0)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        # glEnable(GL_BLEND)
        # glDisable(GL_DEPTH_TEST)

        glBindTexture(GL_TEXTURE_2D, self.texture_cube)
        glPushMatrix()
        glTranslatef(0.0,0.0,-7.0)
        # glRotatef(self.x_axis,1.0,0.0,0.0)
        # glRotatef(0.0,0.0,1.0,0.0)
        # glRotatef(self.z_axis,0.0,0.0,1.0)
        glTranslatef(2*x/self.width, 1 - 2*y/self.height, 0)
#        print(x)
#        print(y)
        glutSolidTeapot(0.5)
        # self.draw_cube()
        glPopMatrix()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        self.x_axis = self.x_axis - 10
        self.z_axis = self.z_axis - 10
    
        glutSwapBuffers()

    def draw_cube(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, -1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, 1.0, 1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, 1.0, 1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, 1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f( 1.0, -1.0, -1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f( 1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, -1.0)
        glEnd()

    def main(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(800, 400)
        glutCreateWindow("OpenGL")
        glutDisplayFunc(self.draw_stuff)
        glutIdleFunc(self.draw_stuff)
        self.init_gl(640, 480)
        glutMainLoop()

a = FromVideo()
a.main()
