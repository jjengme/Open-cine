import numpy as np
from basicfunctions import cine2nparrall, cine2nparr
from pyphantom import Phantom, utils, cine
import matplotlib.pyplot as plt
import cv2 as cv


ph = Phantom() # cine 파일을 불러올 때 항상해줘야됨

fname = 'Z:/03 exp/220126 ilasskorea/2bar 2x/2bar 106.cine'
# fname = 'Z:/03 exp/투명노즐/32uLD.cine'
frame = 0

test = cine2nparr(fname, frame)

# plt.figure
# plt.imshow(test)
# plt.show()

cv.imshow('test',test)
cv.waitKey()
cv.destroyAllWindows()


