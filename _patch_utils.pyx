
cimport cython

import numpy as np
cimport numpy as np
# from libcpp cimport bool

np.import_array()

ctypedef fused numeric:
#    bool
    cython.char
    cython.int
    cython.long
    cython.float
    cython.double

@cython.boundscheck(False)
@cython.wraparound(False)
def _extend_margin(np.ndarray[numeric] arr,
                   tuple patch_shape,
                   str mode,
                   numeric cval):
    pass

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_many_patches2d(np.ndarray[numeric, ndim=3] img,
                        tuple patch_shape,
                        np.ndarray[cython.integral, ndim=2] centers,
                        int step):
    
    cdef int dx = (patch_shape[0] - 1) / step + 1
    cdef int dy = (patch_shape[1] - 1) / step + 1
    
    cdef int hdx = dx//2
    cdef int hdy = dy//2
    
    cdef int p, c, i, j, x, y
    cdef int num_patches = centers.shape[0]
    cdef int img_shape[2]
    for l in xrange(2):
        img_shape[l] = img.shape[l]
    cdef int num_channels = img.shape[2]
    
    cdef int coords[2];
    
    cdef tuple final_patch_shape = (dx, dy)
    
    cdef np.ndarray[numeric, ndim=4] res = np.zeros((num_patches,)+final_patch_shape+(num_channels,), dtype=img.dtype)
    
    with nogil:
        for p in range(num_patches):
            x = centers[p, 0]
            y = centers[p, 1]
            
            for i from -hdx <= i < dx - hdx:
                for j from -hdy <= j < dy - hdy:
                    coords[0] = step * i + x
                    coords[1] = step * j + y
                    
                    for l in xrange(2):
                        if coords[l] < 0:
                            coords[l] = -coords[l]
                        elif coords[l] >= img_shape[l]:
                            coords[l] = img_shape[l] - (coords[l] - img_shape[l] + 1)
                    
                    for c in range(num_channels):
                        res[p, i+hdx, j+hdy, c] = img[coords[0], coords[1], c]
    
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_many_patches3d(np.ndarray[numeric, ndim=4] img,
                        tuple patch_shape,
                        np.ndarray[cython.integral, ndim=2] centers,
                        int step):
    
    cdef int dx = (patch_shape[0] - 1) / step + 1
    cdef int dy = (patch_shape[1] - 1) / step + 1
    cdef int dz = (patch_shape[2] - 1) / step + 1
    
    cdef int hdx = dx//2
    cdef int hdy = dy//2
    cdef int hdz = dz//2
    
    cdef int p, c, i, j, k, x, y, z
    cdef int num_patches = centers.shape[0]
    cdef int img_shape[3]
    for l in xrange(3):
        img_shape[l] = img.shape[l]
    cdef int num_channels = img.shape[3]
    
    cdef int coords[3];
    
    cdef tuple final_patch_shape = (dx, dy, dz)
    
    cdef np.ndarray[numeric, ndim=5] res = np.zeros((num_patches,)+final_patch_shape+(num_channels,), dtype=img.dtype)
    
    with nogil:
        for p in range(num_patches):
            x = centers[p, 0]
            y = centers[p, 1]
            z = centers[p, 2]
            
            for i from -hdx <= i < dx - hdx:
                for j from -hdy <= j < dy - hdy:
                    for k from -hdz <= k < dz - hdz:
                        coords[0] = step * i + x
                        coords[1] = step * j + y
                        coords[2] = step * k + z
                        
                        for l in xrange(3):
                            if coords[l] < 0:
                                coords[l] = -coords[l]
                            elif coords[l] >= img_shape[l]:
                                coords[l] = img_shape[l] - (coords[l] - img_shape[l] + 1)
                        
                        for c in range(num_channels):
                            res[p, i+hdx, j+hdy, k+hdz, c] = img[coords[0], coords[1], coords[2], c]
    
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def _add_many_patches3d(np.ndarray[numeric, ndim=3] img,
                        tuple patch_shape,
                        np.ndarray[cython.integral, ndim=2] centers,
                        np.ndarray[numeric, ndim=4] values
                        ):
    
    cdef int dx = patch_shape[0]
    cdef int dy = patch_shape[1]
    cdef int dz = patch_shape[2]
    
    cdef int hdx = dx//2
    cdef int hdy = dy//2
    cdef int hdz = dz//2
    
    cdef int c, i, j, k, x, y, z
    
    cdef int img_shape[3]
    for l in xrange(3):
        img_shape[l] = img.shape[l]
    cdef int coords[3];
    
    with nogil:
        for c in range(centers.shape[0]):
            x = centers[c, 0]
            y = centers[c, 1]
            z = centers[c, 2]
            
            for i from -hdx <= i < dx - hdx:
                coords[0] = i + x
                if coords[0] < 0 or coords[0] >= img_shape[0]:
                    continue
                
                for j from -hdy <= j < dy - hdy:
                    coords[1] = j + y
                    if coords[1] < 0 or coords[1] >= img_shape[1]:
                        continue
                    
                    for k from -hdz <= k < dz - hdz:
                        coords[2] = k + z
                        if coords[2] < 0 or coords[2] >= img_shape[2]:
                            continue
                        
                        img[coords[0], coords[1], coords[2]] += values[c, i+hdx, j+hdy, k+hdz]
