import numpy as np
import cv2

M = cv2.imread('final_renders/mask.png').astype(np.float32)
R = cv2.imread('final_renders/render_with_obj.png').astype(np.float32)
I = cv2.imread('final_renders/background.png').astype(np.float32)
E = cv2.imread('final_renders/render_without_obj.png').astype(np.float32)
c=4

Rf = cv2.bilateralFilter(R,15,75,75)
Ef = cv2.bilateralFilter(E,15,75,75)

composite = M*R + (1-M)*(I + (Rf-Ef)*c)

cv2.imwrite('final_renders/composite.png', composite)