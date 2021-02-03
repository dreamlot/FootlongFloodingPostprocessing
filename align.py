# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:21:33 2019

@author: ningyu
"""

#from __future__ import print_function
import cv2
import numpy as np
from imshow import imshow
#import imutils
#import ero_dil
#import warnings

    
def alignImage(contourref,contour,img,usemask = None):
    if usemask == None:
        usemask = np.full(img.shape,255,dtype=np.uint8)
    
    # Find size of image1
    sz = contourref.shape


    '''
    First, do translational fix.
    '''
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    #warp_mode = cv2.MOTION_HOMOGRAPHY
     
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 1000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-6;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \
                number_of_iterations,  termination_eps)
    
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (contourref,contour,warp_matrix, warp_mode, criteria, usemask, 5)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (contour, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        result = cv2.warpPerspective (img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(contour, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        result = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
     
    N_ITERATION_OUTER = 6;
    N_ITERATION_START = 500;
    N_ITERATION_END = 10000;
    EPS_START = 1e-6;
    # set ESP_END to larger to decrease run time
    #EPS_END = 1e-11;
    EPS_END = 1e-8;
    for lp1 in range(N_ITERATION_OUTER):
        '''
        Second, do rotational fix.
        '''
        # Define the motion model
        warp_mode = cv2.MOTION_EUCLIDEAN
        #warp_mode = cv2.MOTION_HOMOGRAPHY
         
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
         
        # Specify the number of iterations.
         
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        '''
        if lp1 == N_ITERATION_OUTER - 1:
            number_of_iterations = N_ITERATION_END;
            termination_eps = EPS_END;
        else:
            number_of_iterations = N_ITERATION_START;
            termination_eps = EPS_START;
        '''
        number_of_iterations = int( ( lp1*N_ITERATION_END + (N_ITERATION_OUTER-lp1)*N_ITERATION_START ) / N_ITERATION_OUTER )
        termination_eps =  ( lp1*EPS_END + (N_ITERATION_OUTER-lp1)*EPS_START ) / N_ITERATION_OUTER
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \
                    number_of_iterations,  termination_eps)
        
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (contourref,im2_aligned,warp_matrix, warp_mode, criteria, usemask, 5)
         
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            result = cv2.warpPerspective (result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            result = cv2.warpAffine(result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
         
        
        '''
        Third, do translational fix, again
        '''
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
        #warp_mode = cv2.MOTION_HOMOGRAPHY
         
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
         
        # Specify the number of iterations.
         
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        '''
        if lp1 == N_ITERATION_OUTER - 1:
            number_of_iterations = N_ITERATION_END;
            termination_eps = EPS_END;
        else:
            number_of_iterations = N_ITERATION_START;
            termination_eps = EPS_START;
        '''
        number_of_iterations = int( ( lp1*N_ITERATION_END + (N_ITERATION_OUTER-lp1)*N_ITERATION_START ) / N_ITERATION_OUTER )
        termination_eps =  ( lp1*EPS_END + (N_ITERATION_OUTER-lp1)*EPS_START ) / N_ITERATION_OUTER
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \
                    number_of_iterations,  termination_eps)
        
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (contourref,im2_aligned,warp_matrix, warp_mode, criteria,usemask,5)
         
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            result = cv2.warpPerspective (result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            result = cv2.warpAffine(result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
         
        
    # Show final results
    #imshow(im2_aligned[0],name="Aligned Image 2",showresult=True )

    return( im2_aligned, result )
    
    
    
    
    
    
if __name__ == '__main__':
    print("This file provides the function for aligning two images.")
    
    im1 = cv2.imread("D:\\footlongmodel2\\DSC_0536.jpg")
    im2 = cv2.imread("D:\\footlongmodel2\\DSC_0570.jpg")
    
    im3 = np.uint8(im1[600:1800,3400:4000,1])
    im4 = np.uint8(im2[600:1800,3400:4000,1])
    
    #im3 = np.uint8(im1[:,:,1])
    #im4 = np.uint8(im2[:,:,1])
    
    imshow(im3,name='im3',x=im3.shape[0],y=im3.shape[1])
    imshow(im4,name='im4',x=im4.shape[0],y=im4.shape[1])
    
    im5,res = alignImage(im3,im4,im4)
    im6,res = alignImage(im3,im4,im3)