import numpy as np
cimport numpy as np
cimport cython
np.import_array()

from lib.pysvd3.svd3 import svd3 as svd3
from lib.pysvd3.svd3 import qr3 as qr3
from lib.pysvd3.svd3 import pd3 as pd3

# it is faster to bypass the python wrapper
cdef extern from '../lib/pysvd3/lib/svd3/svd3.h':
    void svd(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float &u11, float &u12, float &u13, \
            float &u21, float &u22, float &u23, \
            float &u31, float &u32, float &u33, \
            float &s11, float &s12, float &s13, \
            float &s21, float &s22, float &s23, \
            float &s31, float &s32, float &s33, \
            float &v11, float &v12, float &v13, \
            float &v21, float &v22, float &v23, \
            float &v31, float &v32, float &v33) nogil
    void multAB(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float b11, float b12, float b13, \
            float b21, float b22, float b23, \
            float b31, float b32, float b33, \
            float &m11, float &m12, float &m13, \
            float &m21, float &m22, float &m23, \
            float &m31, float &m32, float &m33) nogil


cdef extern from 'matop.h':
    void scale_vec(\
            float a1, float a2, float a3, \
            float &r1, float &r2, float &r3, \
            float s) nogil

    void est_scale(\
            float a1, float a2, float a3, \
            float b1, float b2, float b3, \
            float c, float &s) nogil


cdef inline void c_orthogonal_polar_factor(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float &r11, float &r12, float &r13, \
            float &r21, float &r22, float &r23, \
            float &r31, float &r32, float &r33) nogil:

    cdef float u11, u12, u13, u21, u22, u23, u31, u32, u33
    cdef float v11, v12, v13, v21, v22, v23, v31, v32, v33

    svd(\
        a11, a12, a13, a21, a22, a23, a31, a32, a33, \
        u11, u12, u13, u21, u22, u23, u31, u32, u33, \
        r11, r12, r13, r21, r22, r23, r31, r32, r33, \
        v11, v12, v13, v21, v22, v23, v31, v32, v33)

    multAB(\
        u11, u12, u13, u21, u22, u23, u31, u32, u33, \
        v11, v21, v31, v12, v22, v32, v13, v23, v33, \
        r11, r12, r13, r21, r22, r23, r31, r32, r33)


def orthogonal_polar_factor(np.ndarray[float, ndim=2, mode='c'] A not None):
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3x3")

    cdef np.ndarray[float, ndim=2, mode='c'] R = np.zeros((3,3), dtype=np.float32)

    c_orthogonal_polar_factor(
            A[0, 0], A[0, 1], A[0, 2],
            A[1, 0], A[1, 1], A[1, 2],
            A[2, 0], A[2, 1], A[2, 2],
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2])

    return R


def procrutes(np.ndarray[float, ndim=2] X not None,
              np.ndarray[float, ndim=2] Y not None):

    # solve argmin_R ||RX - Y||_F, subject to RTR = I and det(R) = 1
    A = np.matmul(Y, X.T)
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3xN")

    cdef np.ndarray[float, ndim=2, mode='c'] R = np.zeros((3,3), dtype=np.float32)

    c_orthogonal_polar_factor(
            A[0, 0], A[0, 1], A[0, 2],
            A[1, 0], A[1, 1], A[1, 2],
            A[2, 0], A[2, 1], A[2, 2],
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2])

    return R

def anitropic_procrutes(np.ndarray[float, ndim=2] X not None,
                        np.ndarray[float, ndim=2] Y not None,
                        np.ndarray[float, ndim=1] S=None,
                        int iter_num=20):

    # solve argmin_R,S ||RSX - Y||_F, subject to RTR = I, det(R) = 1 and S is
    # diagonal and positive
    A = np.matmul(Y, X.T)
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3xN")

    if S is None:
        S = np.ones((3,), dtype=np.float32)

    cdef float[:] SV = S
    cdef np.ndarray[float, ndim=2, mode='c'] R = np.zeros((3,3), dtype=np.float32)
    cdef float u11, u12, u13, u21, u22, u23, u31, u32, u33

    Xs = np.square(X).sum(-1)

    for _ in range(iter_num):
        scale_vec(A[0, 0], A[1, 0], A[2, 0], u11, u21, u31, SV[0])
        scale_vec(A[0, 1], A[1, 1], A[2, 1], u12, u22, u32, SV[1])
        scale_vec(A[0, 2], A[1, 2], A[2, 2], u13, u23, u33, SV[2])

        c_orthogonal_polar_factor(
                u11, u12, u13,
                u21, u22, u23,
                u31, u32, u33,
                R[0, 0], R[0, 1], R[0, 2],
                R[1, 0], R[1, 1], R[1, 2],
                R[2, 0], R[2, 1], R[2, 2])
        est_scale(
                A[0, 0], A[1, 0], A[2, 0],
                R[0, 0], R[1, 0], R[2, 0],
                Xs[0], SV[0])
        est_scale(
                A[0, 1], A[1, 1], A[2, 1],
                R[0, 1], R[1, 1], R[2, 1],
                Xs[1], SV[1])
        est_scale(
                A[0, 2], A[1, 2], A[2, 2],
                R[0, 2], R[1, 2], R[2, 2],
                Xs[2], SV[2])

    return R, S

