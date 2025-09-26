import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

teste = cv2.imread('Lab 1\Moodle\print.png')

B=teste[:,:,0]
G=teste[:,:,1]
R=teste[:,:,2]

min_blue=np.min(B)
min_green=np.min(G)
min_red=np.min(R)

max_blue=np.max(B)
max_green=np.max(G)
max_red=np.max(R)

print("Range azul: "+str(min_blue)+" - "+str(max_blue))
print("Range verde: "+str(min_green)+" - "+str(max_green))
print("Range vermelho: "+str(min_red)+" - "+str(max_red))