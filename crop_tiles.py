from skimage.transform import resize
import cv2 as cv
import glob
from math import comb
from PIL import Image as PImage
import itertools
import matplotlib.pyplot as plt
import numpy as np

a=np.zeros([2637,2637])
img_size=879
interval=int(a.shape[0]/img_size)
points=np.arange(0,a.shape[0]+1,img_size)

con=np.zeros([9,2]).astype(str)
row=0

for i in range(int(interval**2)):
    
    #i want this to control the first term because this changes per 3 rows while i changes every row 
    #row -1 is used earlier because 0 also goes into this if statement
    if i%3==0:
        row+=1
        column=0
        #print(i)

    for j in range(2):
        
        if j==0:
            con[i,j]=slice(points[row-1],points[row])
            #column+=1
        else:
            #print(i,j)
            #print('col: ', column)
            con[i,j]=slice(points[column],points[column+1])
            
            column+=1
            
        #print(con[i,j])
        #print()
        
cmoney=list(con)
print(con)
#this converts the slice which is a string to the actual product that i want
print(a[eval(con[0][0]),eval(con[0][1])].shape)

    
