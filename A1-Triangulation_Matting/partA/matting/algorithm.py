## CSC320 Winter 2019 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = { 
            'backA': None, 
            'backB': None, 
            'compA': None, 
            'compB': None, 
            'colOut': None,
            'alphaOut': None, 
            'backIn': None, 
            'colIn': None, 
            'alphaIn': None, 
            'compOut': None, 
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self): 
        return {
            'backA':{'msg':'Image filename for Background A Color','default':None},
            'backB':{'msg':'Image filename for Background B Color','default':None},
            'compA':{'msg':'Image filename for Composite A Color','default':None},
            'compB':{'msg':'Image filename for Composite B Color','default':None},
        }
    # Same as above, but for the output arguments
    def mattingOutput(self): 
        return {
            'colOut':{'msg':'Image filename for Object Color','default':['color.tif']},
            'alphaOut':{'msg':'Image filename for Object Alpha','default':['alpha.tif']}
        }
    def compositingInput(self):
        return {
            'colIn':{'msg':'Image filename for Object Color','default':None},
            'alphaIn':{'msg':'Image filename for Object Alpha','default':None},
            'backIn':{'msg':'Image filename for Background Color','default':None},
        }
    def compositingOutput(self):
        return {
            'compOut':{'msg':'Image filename for Composite Color','default':['comp.tif']},
        }
    
    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################
            
    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Image file with file name, {}, is NOT opened successfully!'.format(fileName)

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # read the image first using OpenCV, note that image is read in as BGR instead of RGB
        img = cv.imread(fileName)
        
        # if image exits and read successfully
        if img is not None:
            # convert values into the range of [0, 1] in order to ensure computations later
            # to eliminate wired color issue due to overflow
            img = img / float(255)
            self._images[key] = img
            success = True
            msg = 'Image file with file name, {}, opened and stored successfully!'.format(fileName)

        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63. 
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Result image with file name, {}, is NOT written successfully!'.format(fileName)

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        result_img = self._images[key]

        if result_img is not None:
            result_img = np.rint(result_img * 255)
            isWritten = cv.imwrite(fileName, result_img)
            if isWritten is not False:
                success = True
                msg = 'Result image with file name, {}, is written successfully!'.format(fileName)

        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    def triangulationMatting(self):
        """
        success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Unexpected Error(s) Occurred!'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # NOTE: I will adopt the algorithm shown in Theorem 4 in the "Blue Screen Matting" Paper
        # This paper is the paper that is mentioned in the handout; This formula defined in the
        # paper allows us to use the matrix computation to calculate all alphas and colors together,
        # which produces time efficiency in comparison with the algorithm to use for loop to compute
        # each alpha and color pixel by pixel

        # Check whether all inputs are valid
        backA_img = self._images["backA"]
        backB_img = self._images["backB"]
        compA_img = self._images["compA"]
        compB_img = self._images["compB"]

        if backA_img is None:
            msg = "Invalid input: backA is None!"
            return success, msg

        if backB_img is None:
            msg = "Invalid input: backB is None!"
            return success, msg

        if compA_img is None:
            msg = "Invalid input: compA is None!"
            return success, msg

        if compB_img is None:
            msg = "Invalid input: compB is None!"
            return success, msg

        # # This section is to compute alpha and foreground color map by using for loop
        # # By comparing with reference solution, this algorithm's alpha and col matrices
        # # are more accurate than those generated by using the Theorem 4 algorithm
        # # explained on the paper.
        # height, width = backA_img.shape[0], backA_img.shape[1]
        # alpha_matrix = np.zeros((height, width))
        # color_matrix = np.zeros(backA_img.shape)
        # computation_matrix = np.zeros((6, 4))

        # # Define 1's in the matrix since they are at the fixed positions
        # computation_matrix[0, 0] = 1
        # computation_matrix[1, 1] = 1
        # computation_matrix[2, 2] = 1
        # computation_matrix[3, 0] = 1
        # computation_matrix[4, 1] = 1
        # computation_matrix[5, 2] = 1

        # for i in range (0, height):
        #     for j in range(0, width):
        #         # Read bgr value from compA
        #         bgr_compA = np.reshape(compA_img[i, j], (3,1))

        #         # Read bgr value from compB
        #         bgr_compB = np.reshape(compB_img[i, j], (3,1))

        #         # Read bgr from backA and bgr from backB
        #         bgr_backA = backA_img[i, j]
        #         bgr_backB = backB_img[i, j]

        #         # Set the proper bgr value into computation matrix
        #         computation_matrix[0, 3] = 0 - bgr_backA[0]
        #         computation_matrix[1, 3] = 0 - bgr_backA[1]
        #         computation_matrix[2, 3] = 0 - bgr_backA[2]
        #         computation_matrix[3, 3] = 0 - bgr_backB[0]
        #         computation_matrix[4, 3] = 0 - bgr_backB[1]
        #         computation_matrix[5, 3] = 0 - bgr_backB[2]

        #         # Concatenate bgr from compA and compB
        #         temp1 = np.concatenate((bgr_compA, bgr_compB), axis=0)

        #         # Concatenate bgr from backA and backB
        #         temp2 = np.concatenate( (np.reshape(bgr_backA, (3,1)), 
        #             np.reshape(bgr_backB, (3,1))), axis=0 )

        #         delta_bgr_comp = temp1 - temp2

        #         result_vector = np.dot(np.linalg.pinv(computation_matrix), delta_bgr_comp)
        #         # Set proper values for alpha_matrix and color_matrix
        #         alpha_matrix[i, j] = result_vector[3][0]
        #         color_matrix[i, j] = result_vector[0:3, 0]

        # # Set alphaOut and colOut and reset success and msg
        # self._images["alphaOut"] = np.clip(alpha_matrix, 0, 1)
        # self._images["colOut"] = np.clip(color_matrix, 0, 1)
        # success = True
        # msg = "Triangulation Matting executed successfully!"

        # This section is to compute alpha matrix and color matrix by using matrix arithmetic
        backA_B = backA_img[:,:,0]
        backA_G = backA_img[:,:,1]
        backA_R = backA_img[:,:,2]

        backB_B = backB_img[:,:,0]
        backB_G = backB_img[:,:,1]
        backB_R = backB_img[:,:,2]

        compA_B = compA_img[:,:,0]
        compA_G = compA_img[:,:,1]
        compA_R = compA_img[:,:,2]

        compB_B = compB_img[:,:,0]
        compB_G = compB_img[:,:,1]
        compB_R = compB_img[:,:,2]

        denominator = np.square(backA_R - backB_R) + np.square(backA_G - backB_G) + \
            np.square(backA_B - backB_B)

        # Check whether background are the same --> np.any() is False if all zeros
        if not np.any(denominator):
            msg = "Two background images are the same. Invalid inputs!"
            return success, msg

        numerator = np.multiply(compA_R - compB_R, backA_R - backB_R) + \
            np.multiply(compA_G - compB_G, backA_G - backB_G) + \
            np.multiply(compA_B - compB_B, backA_B - backB_B)

        alpha_matrix = 1 - np.divide(numerator, denominator)

        # The section below is to compute color_matrix
        color_matrix = np.zeros(compA_img.shape)
        color_matrix[:,:,0] = compA_B - np.multiply((1 - alpha_matrix), backA_B)
        color_matrix[:,:,1] = compA_G - np.multiply((1 - alpha_matrix), backA_G)
        color_matrix[:,:,2] = compA_R - np.multiply((1 - alpha_matrix), backA_R)

        # Set alphaOut and colOut and reset success and msg
        self._images["alphaOut"] = np.clip(alpha_matrix, 0, 1)
        # cv.imshow("./try.jpg", self._images["alphaOut"])
        # cv.waitKey(0)
        self._images["colOut"] = np.clip(color_matrix, 0, 1)
        success = True
        msg = "Triangulation Matting executed successfully!"

        #########################################

        return success, msg

        
    def createComposite(self):
        """
        success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Unexpected Error(s) Occurred!'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # Check whether all inputs are valid
        alpha_matrix = self._images["alphaIn"]
        color_matrix = self._images["colIn"]
        back_img = self._images["backIn"]

        if alpha_matrix is None:
            msg = "Invalid input: alphaIn is None!"
            return success, msg

        if color_matrix is None:
            msg = "Invalid input: colIn is None!"
            return success, msg

        if back_img is None:
            msg = "Invalid input: backIn is None!"
            return success, msg

        # Compute composite image, following BGR order, note that BGR values for alpha_matrix are the
        # same -- 3D alpha matrix not 2D here
        result_img = np.zeros(back_img.shape)
        #print result_img[:, 0, :]
        result_img[:,:,0] = np.multiply(alpha_matrix[:,:,0], color_matrix[:,:,0]) + \
            np.multiply((1 - alpha_matrix[:,:,0]), back_img[:,:,0])
        result_img[:,:,1] = np.multiply(alpha_matrix[:,:,1], color_matrix[:,:,1]) + \
            np.multiply((1 - alpha_matrix[:,:,1]), back_img[:,:,1])
        result_img[:,:,2] = np.multiply(alpha_matrix[:,:,2], color_matrix[:,:,2]) + \
            np.multiply((1 - alpha_matrix[:,:,2]), back_img[:,:,2])

        # Set compOut and reset success and msg
        self._images["compOut"] = np.clip(result_img, 0, 1)
        success = True
        msg = "Created composite image successfully!"

        #########################################

        return success, msg


