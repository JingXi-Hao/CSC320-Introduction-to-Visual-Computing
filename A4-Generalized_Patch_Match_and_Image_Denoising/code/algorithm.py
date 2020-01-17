# CSC320 Winter 2019
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################

    # define shape
    k = len(f_heap[0][0])
    N, M = source_patches.shape[0:2]

    # check whether propagation_enabled and random_enabled are both False
    if (not propagation_enabled) and (not random_enabled):
        return global_vars

    # define loops start and end based on the value of odd_iteration
    if odd_iteration:
        row_start = 0
        row_end = N
        column_start = 0
        column_end = M
        step = 1
    else:
        # if even iteration, do a reverse order scan
        row_start = N - 1
        row_end = -1
        column_start = M - 1
        column_end = -1
        step = -1

    # define index difference for later use
    index_diff = (-1) * step

    # define k and search radius list for random search
    if random_enabled:
        search_rads_list = []
        t = 0
        while True:
            search_rad = w * (alpha ** t)
            if search_rad < 1:
                break
            else:
                t += 1
                search_rads_list.append(search_rad)
                search_rads_list.append(search_rad)
        search_rads_list = np.array(search_rads_list).reshape((t, 2))
        search_rads_list = np.repeat(np.array([search_rads_list]), k, axis=0)
        search_rads_list = search_rads_list.flatten().reshape((-1, 2))

    # define functions
    get_second_element = lambda item: item[2]

    # loop over each pixel
    for i in range(row_start, row_end, step):
        for j in range(column_start, column_end, step):

            # define current source patch and current index
            source = source_patches[i, j]
            cur_index = np.array([i, j])
            
            # propagation process
            if propagation_enabled:
                # define variables
                offsets = []
                skip = False
                
                # if (0,0) or the last image pixel, then no need to propagate, but need to do random
                # search if random_enabled is True
                if (i == row_start) and (j == column_start):
                    skip = True

                if skip == False:
                    # when index is valid, add f(x+1,y) and f(x,y+1) for even iteration while
                    # add f(x-1,y) and f(x,y-1) for odd iteration, note that no need to add 
                    # current offset since the distance is already computed and saved in f_heap
                    if i != row_start:
                        cur_offsets = list(map(get_second_element, f_heap[i + index_diff][j]))
                        offsets.extend(cur_offsets)
                        
                    if j != column_start:
                        cur_offsets = list(map(get_second_element, f_heap[i][j + index_diff]))
                        offsets.extend(cur_offsets)
                        
                    # convert into numpy array and offsets would not be an empty numpy array
                    offsets = np.array(offsets)
                    target_index = cur_index + offsets

                    # make sure all target indices found are within the target image indices range
                    target_index[:, 0] = np.clip(target_index[:, 0], 0, N - 1)
                    target_index[:, 1] = np.clip(target_index[:, 1], 0, M - 1)
                    target_index = target_index.astype(int)

                    # redefine offsets in case that some target indices are out of bound
                    offsets = (target_index - cur_index).astype(int)

                    # find all target patches
                    targets = target_patches[target_index[:, 0], target_index[:, 1]]

                    # compute distances -- Euclidean distance but weighted
                    diffs = targets - source # k x C x P^2
                    invalid_mask = np.isnan(diffs)
                    diffs[invalid_mask] = 0
                    valid_mask = (np.logical_not(invalid_mask)) / float(1) # k x C x P^2
                    weights = np.sum(np.sum(valid_mask, axis=-1), axis=-1) # k
                    diffs = np.sum(np.sum(diffs ** 2, axis=-1), axis=-1) # k
                    diffs = np.divide(diffs, weights) # k

                    # update f_heap and f_coord_dictionary
                    for index in range(len(diffs)):
                        max_diff = (-1) * f_heap[i][j][0][0]
                        diff_value = diffs[index]
                        offset_value = offsets[index]
                        # second condition is to eliminate duplicates
                        updateFlag = (diff_value < max_diff) and (tuple(offset_value) not in f_coord_dictionary[i][j].keys())
                        if updateFlag:
                            count = next(_tiebreaker)
                            heappushpop(f_heap[i][j], ((-1) * diff_value, count, offset_value))
                            f_coord_dictionary[i][j][tuple(offset_value)] = (-1) * diff_value

            # random search process
            if random_enabled:
                # compute offsets
                R_k = np.random.uniform(-1, 1, (k, t, 2)).reshape((-1, 2))
                cur_offsets = np.repeat(np.array(map(get_second_element, f_heap[i][j])), t, axis=0)
                # print("cur_offsets: {}".format(cur_offsets.shape))
                U_k = cur_offsets + search_rads_list * R_k

                # compute target index
                target_index = (U_k + cur_index).astype(int)

                # make sure all target indices found are within the target image indices range
                target_index[:, 0] = np.clip(target_index[:, 0], 0, N - 1)
                target_index[:, 1] = np.clip(target_index[:, 1], 0, M - 1)
                target_index = target_index.astype(int)

                # redefine offsets
                offsets = (target_index - cur_index).astype(int)

                # find all target patches
                targets = target_patches[target_index[:, 0], target_index[:, 1]]

                # compute distances -- Euclidean distance but weighted
                diffs = targets - source # k x C x P^2
                invalid_mask = np.isnan(diffs)
                diffs[invalid_mask] = 0
                valid_mask = (np.logical_not(invalid_mask)) / float(1) # k x C x P^2
                weights = np.sum(np.sum(valid_mask, axis=-1), axis=-1) # k
                diffs = np.sum(np.sum(diffs ** 2, axis=-1), axis=-1) # k
                diffs = np.divide(diffs, weights) # k

                # update f_heap and f_coord_dictionary
                for index in range(len(diffs)):
                    max_diff = (-1) * f_heap[i][j][0][0]
                    diff_value = diffs[index]
                    offset_value = offsets[index]
                    # second condition is to eliminate duplicates
                    updateFlag = (diff_value < max_diff) and (tuple(offset_value) not in f_coord_dictionary[i][j].keys())
                    if updateFlag:
                        count = next(_tiebreaker)
                        heappushpop(f_heap[i][j], ((-1) * diff_value, count, offset_value))
                        f_coord_dictionary[i][j][tuple(offset_value)] = (-1) * diff_value

    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # define variables
    k, N, M = f_k.shape[0:3]
    source_index = make_coordinates_matrix((N, M))

    # find all target indices
    target_index = f_k + source_index # k x N x M x 2

    # find invalid index condition -- mask
    condition = (target_index[:, :, :, 0] >= 0) * (target_index[:, :, :, 0] < N) * (target_index[:, :, :, 1] >= 0) * (target_index[:, :, :, 1] < M)
    condition = np.logical_not(condition) # k x N x M

    # clip target indices within its valid range
    target_index[:, :, :, 0] = np.clip(target_index[:, :, :, 0], 0, N - 1)
    target_index[:, :, :, 1] = np.clip(target_index[:, :, :, 1], 0, M - 1)
    target_index = target_index.astype(int)

    # find all target patches
    targets = target_patches[target_index[:, :, :, 0], target_index[:, :, :, 1]]

    # compute distances -- Euclidean distance but weighted
    diffs = targets - source_patches # k x N x M x C x P^2
    invalid_mask = np.isnan(diffs)
    diffs[invalid_mask] = 0
    valid_mask = (np.logical_not(invalid_mask)) / float(1) # k x N x M x C x P^2
    weights = np.sum(np.sum(valid_mask, axis=-1), axis=-1) # k x N x M
    diffs = np.sum(np.sum(diffs ** 2, axis=-1), axis=-1) # k x N x M
    diffs = np.divide(diffs, weights) # k x N x M

    # set invalid target indices to with distance = np.inf and times negative one to diffs
    diffs[condition] = np.inf
    diffs = (-1) * diffs

    # define f_heap and f_coord_dictionary
    f_heap = []
    f_coord_dictionary = []

    # update f_heap and f_coord_dictionary
    for i in range(N):
        row_heap_list = []
        row_dict_list = []

        for j in range(M):
            pixel_heap = []
            pixel_dict = {}

            for s in range(k):
                count = next(_tiebreaker)
                heappush(pixel_heap, (diffs[s, i, j], count, f_k[s, i, j]))
                pixel_dict[tuple(f_k[s, i, j])] = diffs[s, i, j]

            row_heap_list.append(pixel_heap)
            row_dict_list.append(pixel_dict)
        f_heap.append(row_heap_list)
        f_coord_dictionary.append(row_dict_list)

    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # define shape
    k = len(f_heap[0][0])
    N = len(f_heap)
    M = len(f_heap[0])

    # initialize
    f_k = np.zeros((k, N, M, 2))
    D_k = np.zeros((k, N, M))

    # define functions
    get_first_element = lambda item: (-1) * item[0]
    get_last_elememt = lambda item: item[2]

    # update the f_k and D_k
    for i in range(N):
        for j in range(M):
            # need to be sorted here, use nlargest to sort by distance here
            cur_heap = nlargest(k, f_heap[i][j])
            distances = np.array(map(get_first_element, cur_heap))
            offsets = np.array(map(get_last_elememt, cur_heap)).astype(int)
            D_k[:, i, j] = distances
            f_k[:, i, j] = offsets

    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # define shape
    N, M = target.shape[0:2]
    last = target.shape[-1]
    k = len(f_heap[0][0])

    # define source indices, nnf and similarity, and target indices
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)
    source_index = make_coordinates_matrix((N, M))
    target_index = source_index + f_k

    # clip target indices within its valid range
    target_index[:, :, :, 0] = np.clip(target_index[:, :, :, 0], 0, N - 1)
    target_index[:, :, :, 1] = np.clip(target_index[:, :, :, 1], 0, M - 1)
    target_index = target_index.astype(int)

    # define weights
    weights = np.exp((-1 * (D_k ** 0.5)) / (h ** 2)) # k x N x M
    Z = np.sum(weights, axis=0) # N x M (size for Z)
    normalized_weights = np.divide(weights, Z) # k x N x M (size for weights matrix)
    normalized_weights = np.repeat(normalized_weights.flatten(), last).reshape((k, N, M, last))

    # define denoised
    targets = target[target_index[:, :, :, 0], target_index[:, :, :, 1]] # k x N x M x 3
    denoised = np.sum(targets * normalized_weights, axis=0) # sum over k

    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################

    # initialize all coordinates
    init_index = make_coordinates_matrix(target.shape)

    # compute target indices and make sure they all within valid range
    target_index = init_index + f
    target_index[:, :, 0] = np.clip(target_index[:, :, 0], 0, target.shape[0] - 1)
    target_index[:, :, 1] = np.clip(target_index[:, :, 1], 0, target.shape[1] - 1)
    target_index = target_index.astype(int)

    # rearrange target image in order to reconstruct the source image
    rec_source = target[target_index[:, :, 0], target_index[:, :, 1]]

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
