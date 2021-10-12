from skimage.transform import resize
import os
import argparse
import cv2 as cv
import glob
from math import comb
from PIL import Image as PImage
import itertools
import matplotlib.pyplot as plt
import numpy as np

parser= argparse.ArgumentParser(description='Takes in a tiled image and separates them into its individual tiles. This currently only works with square images (3x3,5x5,etc.)')
parser.add_argument("--img_dir")
parser.add_argument("--tiles",type=int)
parser.add_argument("--output_dir")
args=parser.parse_args()

a=np.zeros([2637,2637])
img_size=879


def get_tile_slices(img,tiles):

    row=0

    img_size=img.shape[0]/np.sqrt(tiles)

    slices=np.zeros([tiles,2]).astype(str)

    interval=int(img.shape[0]/img_size)
    points=np.arange(0,img.shape[0]+1,img_size,dtype=int)

    for i in range(int(interval**2)):
        
        #i want this to control the first term because this changes per 3 rows while i changes every row 
        #row -1 is used earlier because 0 also goes into this if statement
        if i%3==0:
            row+=1
            column=0
            #print(i)

        for j in range(2):
            
            if j==0:
                slices[i,j]=slice(points[row-1],points[row])
                #column+=1
            else:
                #print(i,j)
                #print('col: ', column)
                slices[i,j]=slice(points[column],points[column+1])
                
                column+=1
                
            #print(slices[i,j])
            #print()
            
    return slices

def resize_img(img):
    return  resize(img, (img.shape[0]+1,img.shape[1]+1), preserve_range=True, anti_aliasing=False,order=0).astype(np.int)


first=True

for img in os.listdir(args.img_dir):

    filename=img.split('.')[0]
    img=cv.imread(args.img_dir+img,-1)
    img=resize_img(img)

    if first==True:
        slices=get_tile_slices(img,args.tiles)

        first=False

    for tile in range(args.tiles):
        single_tile=img[eval(slices[tile][0]),eval(slices[tile][1])]
        cv.imwrite(args.output_dir+filename+'_tile'+str(tile)+'.tif',single_tile.astype(np.uint16))

    

    
#this converts the slice which is a string to the actual product that i want
#print(a[eval(slices[0][0]),eval(slices[0][1])].shape)

