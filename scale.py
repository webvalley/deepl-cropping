import numpy as np
from cv2 import resize

def scale(img, dim):    
    return resize(img, (dim[1], dim[0]))