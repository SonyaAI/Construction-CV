#!/usr/bin/env python

#-----------------------------------------------
#--------------- Roof Recognition --------------
#------------------- Onduline ------------------
#-----------------------------------------------
#Author: Sofya Akhmametyeva 
#Date: March 19, 2017 

'''
This file contains the following:
    -Function for drawing key-points and matches between two gray scale images.

Code Requirements:
    Python 2.7.x
    NumPy for Python 2.7.x
    SciPy 
    Matplotlib
    OpenCV 2.4.x
'''


import scipy.io as sio
import numpy as np
from math import pi, sin, cos
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import argparse
import cv2
import os
import time
import matlab.engine #MATLAB integration
import tensorflow as tf 


def get_filenames(folder):
    # returns a list of file names (with extension, without full path) of all files in folder path
    fileNames = []
    for name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, name)):
            fileNames.append(name)
    return fileNames 


def main():
    '''
    Main function contains the image processing and computer vision steps in order to isolate the roof.
    '''
    #A full path to the folders with images of houses with roofs
    imgL1_folder = os.getcwd() + '/data/roofs_level1'   
    imgL2_folder = os.getcwd() + '/data/roofs_level2'  
    imgL3_folder = os.getcwd() + '/data/roofs_level3'  
    imgL4_folder = os.getcwd() + '/data/roofs_level4'  

    filesL1 = get_filenames(imgL1_folder) #file names (Level 1 roofs)
    filesL2 = get_filenames(imgL2_folder) #file names (Level 2 roofs)
    filesL3 = get_filenames(imgL3_folder) #file names (Level 3 roofs)
    filesL4 = get_filenames(imgL4_folder) #file names (Level 4 roofs)


    for filename in filesL1: #each file in a folder

        #Preprocessing--------------------------------------------------------

        #print type(filename) # str type 
        img_file = os.path.join(imgL1_folder, filename)
        image = cv2.imread(img_file)
        height, width = image.shape[:2]
        print 'height = ' + str(height) + ', width = ' + str(width)

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        b = 5 # blur block size
        blur_gray = cv2.blur(gray, (b, b))

        #input into thresholding should be a grey scale image
        thresh_adaptmean = cv2.adaptiveThreshold(blur_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        thresh_adaptgaus = cv2.adaptiveThreshold(blur_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        ret, thresh = cv2.threshold(blur_gray, 127, 255, 0)

        #find edges 
        edges_gray = cv2.Canny(gray,100,200)
        edges_threshAdaptMean = cv2.Canny(thresh_adaptmean,100,200)
        edges_threshAdaptGaus = cv2.Canny(thresh_adaptgaus,100,200)

        #find all contours
        #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #draw contours 

        #Find all straight lines 
        img_lines = image
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        minLineLength = 100 #Minimum length of line. Line segments shorter than this are rejected.
        maxLineGap = 30 #Maximum allowed gap between line segments to treat them as single line.
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(img_lines,(x1,y1),(x2,y2),(0,255,0),2)


        #TensorFlow 
        node1 = tf.constant(3.0, tf.float32)
        node2 = tf.constant(4.0)
        node3 = tf.add(node1,node2)
        sess = tf.Session() #run enough of computational graph to evaluate nodes
        print(sess.run([node1, node2]))
        print(sess.run(node3))

        #cv2.imshow('Original',image)
        #cv2.imshow('Thresh',thresh)
        #cv2.imshow('Thresh adapt mean',thresh_adaptmean)
        #cv2.imshow('Thresh adapt gaus',thresh_adaptgaus)
        #cv2.imshow('Thresh',edges_gray)
        #cv2.imshow('Edges thresh adapt mean',edges_threshAdaptMean)
        #cv2.imshow('Edges thresh adapt gaus',edges_threshAdaptGaus)
        cv2.imshow('Roof - Lines',img_lines)
        cv2.waitKey(0) #wait for a key stroke
        cv2.destroyAllWindows()        



if __name__ == "__main__":
    print("roof_recog.py is being run directly")
    main()
else:
    print("roof_recog.py is being imported into another module")