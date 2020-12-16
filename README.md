## ES143 Final Project: Team Super Super Spicy
### Augmented Reality with Facial Feature Detection and Tracking

Nila Annadurai, Jenny Gu, Ethan Schumann and Sarah Yoon

---

#### Installation Instructions
Run the following commands in your terminal. Note that this project uses Python 3.
```
$ pip install opencv-python
$ pip install numpy
$ brew install cmake
$ pip install dlib
$ pip install imutils
$ pip install Pillow
$ pip install pygame
$ pip install PyOpenGL PyOpenGL_accelerate
```

#### Project Components
1. To view the original OpenCV rendering of AR accessories run `$ python cv_render.py`.
2. To view the finger counting extension run `$ python finger_count.py`.
3. To view the OpenGL rendering (also includes tracking and hand detection), run the following:
```
$ python hand_training.py
$ python hand_gl.py
```
*Note: When running the OpenGL application you can toggle whether or not facial feature rectangles are visible by 
setting the constant `FACE_RECTS` in `hand_gl.py` (line 30) to `True` or `False` (default is `True`). 
Similarly, you can toggle whether rendering the sunglasses depends on hand detection by setting the constant 
`HAND_INTEGRATION` (line 33, default is `True`).*
