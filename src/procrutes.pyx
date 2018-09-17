import numpy as np
cimport numpy as np
cimport cython
np.import_array()

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


    void c_orthogonal_polar_factor(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float &r11, float &r12, float &r13, \
            float &r21, float &r22, float &r23, \
            float &r31, float &r32, float &r33) nogil


    void c_orthogonal_polar_factor_with_diagonal_sum(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float &r11, float &r12, float &r13, \
            float &r21, float &r22, float &r23, \
            float &r31, float &r32, float &r33, float &s) nogil


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


def isotropic_procrutes(np.ndarray[float, ndim=2] X not None,
                        np.ndarray[float, ndim=2] Y not None):

    # solve argmin_R ||RsX - Y||_F, subject to RTR = I, det(R) = 1 and s is a scalar
    A = np.matmul(Y, X.T)
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3xN")

    cdef np.ndarray[float, ndim=2, mode='c'] R = np.zeros((3,3), dtype=np.float32)
    B = np.matmul(X, X.T)
    cdef float s, x_square = B[0, 0] + B[1, 1] + B[2, 2]

    c_orthogonal_polar_factor_with_diagonal_sum(
            A[0, 0], A[0, 1], A[0, 2],
            A[1, 0], A[1, 1], A[1, 2],
            A[2, 0], A[2, 1], A[2, 2],
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2], s)

    s /= x_square

    return R, s


def anisotropic_procrutes(np.ndarray[float, ndim=2] X not None,
                        np.ndarray[float, ndim=2] Y not None,
                        np.ndarray[float, ndim=1] S=None,
                        int iter_num=30):

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


def np_orthogonal_polar_factor(np.ndarray[float, ndim=2, mode='c'] A not None):
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3x3")

    U, S, V = np.linalg.svd(A)
    s = np.sign(np.linalg.det(np.matmul(U, V)))
    R = np.matmul(U*np.array([1, 1, s], dtype=np.float32), V)

    return R


def np_orthogonal_polar_factor_with_diagonal_sum(
        np.ndarray[float, ndim=2, mode='c'] A not None):
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3x3")

    U, S, V = np.linalg.svd(A)
    s = np.sign(np.linalg.det(np.matmul(U, V)))
    R = np.matmul(U*np.array([1, 1, s], dtype=np.float32), V)
    s = np.sum(S)

    return R, s


def np_procrutes(np.ndarray[float, ndim=2] X not None,
                 np.ndarray[float, ndim=2] Y not None):

    A = np.matmul(Y, X.T)
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3xN")

    R = np_orthogonal_polar_factor(A)

    return R


def np_isotropic_procrutes(np.ndarray[float, ndim=2] X not None,
                 np.ndarray[float, ndim=2] Y not None):

    A = np.matmul(Y, X.T)
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3xN")

    R, s = np_orthogonal_polar_factor_with_diagonal_sum(A)
    s /= np.power(X, 2).sum()

    return R, s


def np_anisotropic_procrutes(np.ndarray[float, ndim=2] X not None,
                        np.ndarray[float, ndim=2] Y not None,
                        np.ndarray[float, ndim=1] S=None,
                        int iter_num=30):

    # solve argmin_R,S ||RSX - Y||_F, subject to RTR = I, det(R) = 1 and S is
    # diagonal and positive
    A = np.matmul(Y, X.T)
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Expecting inputs of the size 3xN")

    if S is None:
        S = np.ones((3,), dtype=np.float32)

    Xs = np.square(X).sum(-1)
    non_planar = Xs != 0

    for _ in range(iter_num):
        R = np_orthogonal_polar_factor(A*S)
        S[non_planar] = np.abs((A*R).sum(0)[non_planar] / Xs[non_planar])

    return R, S
