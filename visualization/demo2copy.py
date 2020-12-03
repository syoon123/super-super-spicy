from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image

class AR:
    def __init__(self):
        self.texture_background = None

    def start_gl(self, Width, Height):
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


    def background_texture(self):
        image = Image.open("img.bmp")
        ix = image.size[0]
        iy = image.size[1]
        image = image.tobytes("raw", "RGBX", 0, 1)

        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)

    def draw_background(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 4.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(4.0, -3.0, 4.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(4.0, 3.0, 4.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0, 3.0, 4.0)
        glEnd()

    def _draw_scene(self):
        self.background_texture()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        glTranslatef(0.0,0.0,-10)
        self.draw_background()
        glPopMatrix()
        
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        glEnable(GL_DEPTH_TEST)
        
        glClear(GL_DEPTH_BUFFER_BIT)
        glMaterialfv(GL_FRONT,GL_DIFFUSE,[1.0,0.0,0.0,0.0])

        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        glRotatef(34.0,0.0,1.0,0.0)
        glRotatef(20.0,0.0,0.0,1.0)
        glutSolidTeapot(1.0)
        glPopMatrix()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
    
        glutSwapBuffers()
        

    def ar(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(800, 400)
        glutCreateWindow("OpenGL")
        glutDisplayFunc(self._draw_scene)
        self.start_gl(640, 480)
        glutMainLoop()

test = AR()
test.ar()

