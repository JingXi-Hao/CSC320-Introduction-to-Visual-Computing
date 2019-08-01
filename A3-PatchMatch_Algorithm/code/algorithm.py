# CSC320 Winter 2019
# Assignment 3
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

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
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
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
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
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure

def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # define variables
    rows, columns = source_patches.shape[0 : 2]
    t_rows, t_columns = target_patches.shape[0 : 2]

    # initialize best_D if it is None:
    if best_D is None:
        # compute target indices
        source_index = make_coordinates_matrix((rows, columns))
        target_index = source_index + new_f

        # make sure target indices computed are within the target image index range
        target_index[:, :, 0] = np.clip(target_index[:, :, 0], 0, t_rows - 1)
        target_index[:, :, 1] = np.clip(target_index[:, :, 1], 0, t_columns - 1)

        # rearrange the target patches as the coordinates computed -- where each source
        # patch matches its target match, then compute difference for each pair
        patches = target_patches[target_index[:, :, 0], target_index[:, :, 1]]
        diffs = source_patches - patches

        # penalize the NaN caused by target patches and ignore the NaN caused by source patches,
        # this way could reduce the value of the sum (reduce the risk of overflow)
        # note that square root is quite expensive, thus NOT compute it
        # HERE: penalize every NaN with 255 for efficiency -- two methods produce same results
        # thus, pick the more efficient one
        # diffs[np.isnan(source_patches)] = 0
        diffs[np.isnan(diffs)] = 255
        best_D = np.sum(np.sum(diffs ** 2, axis=3), axis=2)
        # best_D[np.isnan(best_D)] = float('inf')
        # print(np.argwhere(np.isnan(best_D)))

    # check whether propagation_enabled and random_enabled are both False
    if (not propagation_enabled) and (not random_enabled):
        return new_f, best_D, global_vars

    # define loops start and end based on the value of odd_iteration
    if odd_iteration:
        row_start = 0
        row_end = rows
        column_start = 0
        column_end = columns
        step = 1
    else:
        # if even iteration, do a reverse order scan
        row_start = rows - 1
        row_end = -1
        column_start = columns - 1
        column_end = -1
        step = -1

    # define index difference for later use
    index_diff = (-1) * step

    # define k and search radius list for random search
    if random_enabled:
        search_rads_list = []
        k = 0
        while True:
            search_rad = w * (alpha ** k)
            if search_rad < 1:
                break
            else:
                k += 1
                search_rads_list.append(search_rad)
                search_rads_list.append(search_rad)
        search_rads_list = np.array(search_rads_list).reshape((k, 2))
        # this way is slow
        # search_rads_list = np.array(search_rads_list).reshape( (len(search_rads_list), 1) )
        # search_rads_list = np.concatenate([search_rads_list, search_rads_list], axis=1)

    # loop over each pixel
    for i in range(row_start, row_end, step):
        for j in range(column_start, column_end, step):

            # define current source patch and current index
            source = source_patches[i, j]
            cur_index = np.array([i, j])
            
            # propagation process
            if propagation_enabled:
                # define variables
                target_index = []
                skip = False
                
                # if (0,0) or the last image pixel, then no need to propagate, but need to do random
                # search if random_enabled is True
                if (i == row_start) and (j == column_start):
                    skip = True

                if skip == False:
                    # when index is valid, add f(x+1,y) and f(x,y+1) for even iteration while
                    # add f(x-1,y) and f(x,y-1) for odd iteration, note that no need to add 
                    # current offset since the distance is already computed and saved in best_D
                    if i != row_start:
                        target_index.append(cur_index + new_f[i + index_diff, j])
                        # new_index = np.array([ cur_index + new_f[i + index_diff, j] ])
                        # target_index = np.concatenate([target_index, new_index], axis=0)
                        
                    if j != column_start:
                        target_index.append(cur_index + new_f[i, j + index_diff])
                        # new_index = np.array([ cur_index + new_f[i, j + index_diff] ])
                        # target_index = np.concatenate([target_index, new_index], axis=0)

                    # convert into numpy array and offsets would not be an empty numpy array
                    # should have one or two neighbors
                    target_index = np.array(target_index)

                    # make sure all target indices found are within the target image indices range
                    # target_index[target_index < 0] = 0
                    # target_index[:, 0][ target_index[:, 0] > t_rows - 1 ] = t_rows - 1
                    # target_index[:, 1][ target_index[:, 1] > t_columns - 1 ] = t_columns - 1
                    target_index[:, 0] = np.clip(target_index[:, 0], 0, t_rows - 1)
                    target_index[:, 1] = np.clip(target_index[:, 1], 0, t_columns - 1)

                    # find all target patches
                    targets = target_patches[target_index[:, 0], target_index[:, 1]]

                    # repeat source for k times for penalization ignorance -- cause inefficiency
                    # sources = np.repeat( np.array([source]), target_index.shape[0], axis=0)

                    # penalize the NaN caused by target patches and ignore the NaN caused by source patches
                    # this way could reduce the value of the sum (reduce the risk of overflow)
                    # note that square root is quite expensive, thus NOT compute it
                    # HERE: penalize every NaN with 255 for efficiency -- two methods produce same results
                    # thus, pick the more efficient one
                    diffs = targets - source
                    # diffs[np.isnan(sources)] = 0
                    diffs[np.isnan(diffs)] = 255
                    diffs = np.sum(np.sum(diffs ** 2, axis=2), axis=1)

                    # update best_D and f
                    cur_bestD = best_D[i, j]
                    min_index = np.argmin(diffs)
                    min_value = diffs[min_index]
                    if min_value < cur_bestD:
                        best_D[i, j] = min_value
                        new_f[i, j] = target_index[min_index] - cur_index

            # random search process
            if random_enabled:
                # compute offsets
                R_k = np.random.uniform(-1, 1, 2 * k).reshape((k, 2))
                U_k = new_f[i, j] + search_rads_list * R_k

                # compute target index
                target_index = (U_k + cur_index).astype(int)

                # make sure all target indices found are within the target image indices range
                target_index[:, 0] = np.clip(target_index[:, 0], 0, t_rows - 1)
                target_index[:, 1] = np.clip(target_index[:, 1], 0, t_columns - 1)

                # find all target patches
                targets = target_patches[target_index[:, 0], target_index[:, 1]]

                # repeat source for k times for penalization ignorance -- cause inefficiency
                # sources = np.repeat( np.array([source]), k, axis=0)

                # penalize the NaN caused by target patches and ignore the NaN caused by source patches
                # this way could reduce the value of the sum (reduce the risk of overflow)
                # note that square root is quite expensive, thus NOT compute it
                # HERE: penalize every NaN with 255 for efficiency -- two methods produce same results
                # thus, pick the more efficient one
                diffs = targets - source
                # diffs[np.isnan(sources)] = 0
                diffs[np.isnan(diffs)] = 255
                diffs = np.sum(np.sum(diffs ** 2, axis=2), axis=1)

                # update best_D and f
                cur_bestD = best_D[i, j]
                min_index = np.argmin(diffs)
                min_value = diffs[min_index]
                if min_value < cur_bestD:
                    offsets = target_index - cur_index
                    best_D[i, j] = min_value
                    new_f[i, j] = offsets[min_index]
                    
    #############################################

    return new_f, best_D, global_vars


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

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # initialize all coordinates
    init_index = make_coordinates_matrix(target.shape)

    # compute target indices and make sure they all within valid range
    target_index = init_index + f
    target_index[:, :, 0] = np.clip(target_index[:, :, 0], 0, target.shape[0] - 1)
    target_index[:, :, 1] = np.clip(target_index[:, :, 1], 0, target.shape[1] - 1)

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
