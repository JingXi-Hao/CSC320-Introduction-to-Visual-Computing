## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################

#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace this dummy value with your own code
    # C = 1
    
    # get coords and w
    coords = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()

    # get the filled value and its mask that shows filled and unfilled pixels
    # inside the patch
    patchFilled, patchInImage = copyutils.getWindow(filledImage, coords, w)
    patchFilled = patchFilled / float(255)

    # get the confidence values for each pixel inside the patch
    patchConf = copyutils.getWindow(confidenceImage, coords, w)[0]
    
    # compute numerator -- the sum of the filled pixels' confidence value
    num = np.sum(patchFilled * patchConf * patchInImage)

    # compute denominator -- the area of the patch inside the image boundary
    den = np.count_nonzero(patchInImage)

    # compute C
    C = np.divide(num, den)

    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace these dummy values with your own code
    # Dy = 1
    # Dx = 0 

    # get coords and w
    coords = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()

    # # this way is very slow
    # imgGray = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)
    # validGray = imgGray * ( filledImage / 255 )
    # patchGray = copyutils.getWindow(validGray, coords, w)[0]

    # # compute gradients use sobel (scharr)
    # gradientsX = cv.Sobel(patchGray, cv.CV_64F, 1, 0, ksize=-1)
    # gradientsY = cv.Sobel(patchGray, cv.CV_64F, 0, 1, ksize=-1)

    # extract a larger patch in order to ensure the accuracy for the
    # border parts of the original size patch
    imgPatch = copyutils.getWindow(inpaintedImage, coords, w+2)[0]
    patchGray = cv.cvtColor(imgPatch, cv.COLOR_BGR2GRAY)

    # compute gradients, ksize=5 produces better results than ksize=3 and ksize=7 does
    # therefore, use ksize=5
    row = 2
    column = patchGray.shape[1] - 2
    gradientsX = cv.Sobel(patchGray, cv.CV_64F, 1, 0, ksize=5)[row:column, row:column]
    gradientsY = cv.Sobel(patchGray, cv.CV_64F, 0, 1, ksize=5)[row:column, row:column]

    # get mask for filled and inside image boundary pixel
    patchFilled, patchInImage = copyutils.getWindow(filledImage, coords, w)
    patchFilled = patchFilled / float(255)
    validMask = np.multiply(patchFilled, patchInImage)

    # apply validMask to x and y gradients
    validGX = np.multiply(gradientsX, validMask)
    validGY = np.multiply(gradientsY, validMask)

    # compute squared magnitude of gradient for each pixel (inside image) in patch
    magnitudes = np.add(validGX ** 2, validGY ** 2)

    # find the coordinates of the maxial elements
    ind = np.unravel_index(np.argmax(magnitudes, axis=None), magnitudes.shape)

    # set Dy and Dx
    Dx = validGX[ind]
    Dy = validGY[ind]

    #########################################
    
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace these dummy values with your own code
    # Ny = 0
    # Nx = 1 

    # get coords and w of the patch in source image
    coords = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()

    # get masks in patch size
    patchFilled = copyutils.getWindow(filledImage, coords, w)[0]
    patchFront = copyutils.getWindow(fillFront, coords, w)[0]

    # if the fill front consists of exactly one pixel, the fill front is degenerate
    # and has no well-defined normal
    if np.count_nonzero(patchFront) == 1:
        Nx = None
        Ny = None
        return Ny, Nx

    # compute gradients at patch center, still ksize=5 works better
    centerGX = cv.Sobel(patchFilled, cv.CV_64F, 1, 0, ksize=5)[w, w]
    centerGY = cv.Sobel(patchFilled, cv.CV_64F, 0, 1, ksize=5)[w, w]

    # compute magnitude for tangent at patch center
    magnitude = np.sqrt( np.add(centerGX ** 2, centerGY ** 2) )

    # set Ny and Nx
    Nx = - centerGX
    Ny = centerGY
    if magnitude != 0:
        Nx = Nx / float(magnitude)
        Ny = Ny / float(magnitude)

    #########################################

    return Ny, Nx

    
