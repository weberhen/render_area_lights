import numpy as np
import cv2

M = cv2.imread('final_renders/mask.png').astype(np.float32)
R = cv2.imread('final_renders/render_with_obj.png').astype(np.float32)
I = cv2.imread('final_renders/background.png').astype(np.float32)
E = cv2.imread('final_renders/render_without_obj.png').astype(np.float32)
c=5

Rf = cv2.bilateralFilter(R,15,75,75)
Ef = cv2.bilateralFilter(E,15,75,75)
cv2.imwrite('Rf.png', Rf)
cv2.imwrite('Ef.png', Ef)
kernel = np.ones((3,3),np.uint8)
M = cv2.dilate(M,kernel,iterations = 1)

composite = M*R + (1-M)*(I/1.2 + (Rf-Ef)*c)#+(1-M)*(Rf-Ef)*c)

cv2.imwrite('final_renders/composite.png', composite)