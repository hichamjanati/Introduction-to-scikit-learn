# -*- coding: utf-8 -*-

############################################################################
############ Code qui positionne les points sur un screenshot #############
############################################################################

import numpy as np
import pandas as pd
import os
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

################ The general home directory has to be ajusted 
Home         = "HOME DIRECTORY WITH DATA FILE (with the relevant screenshot)"
# The following directories don't have to be changed
NewHome      = Home+'/Restructured Data'
NewData      = NewHome + '/New Data'
NewLandmarks = NewHome + '/New Landmarks'
Frames       = Home +'/Frames' 

# If the directory which is supposed to contain the new frames doesn't exist, let's create it:
if not os.path.exists(Frames):
    os.mkdirs(Frames)


# Positioning landmarks on the given screenshot
d    = pd.read_csv(NewData + "/innerData.txt",sep=',')
file = '020e95e0-8a56-4038-87e6-faf484c2802a'

d    = d[d.File=='/'+file+'-landmarks.txt'] 
d    = d.ix[:,"x1":"Time"]

d.Time = d.Time.apply(lambda e: e+1)

img_path      = Frames +'/'+file+'/scene00076.png'
img_path_save = img_path.split('.')[0]+'-mod.png'

img  = Image.open(img_path)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('Arial',7)

row = d[d.Time==76]
row = row.drop(["Time"],axis=1)
row = np.ravel(row).reshape(22,2)

for i,(x,y) in enumerate(row):
    draw.text((x,y),str(i+1),(255,255,255),font=font)

img.save(img_path_save)

imgFile = cv2.imread(img_path_save)

cv2.imshow('photoooo',imgFile)
cv2.waitKey(0)
cv2.destroyAllWindows()
