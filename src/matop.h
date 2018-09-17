/*
 * =====================================================================================
 *
 *       Filename:  matop.h
 *
 *    Description:  matrix operations for procrutes problem
 *
 *        Version:  1.0
 *        Created:  09/14/2018 11:01:21 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  derleeG, 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef MATOP_H
#define MATOP_H
#include "../lib/pysvd3/lib/svd3/svd3.h"
// scale a 3 vector
inline void scale_vec(
        float a1, float a2, float a3,
        float &r1, float &r2, float &r3,
        float s)
{
    r1=a1*s; r2=a2*s; r3=a3*s;
}

//
inline void est_scale(
        float a1, float a2, float a3,
        float b1, float b2, float b3,
        float c, float &s)
{
    s = c==0 ? s : fabsf((a1*b1 + a2*b2 + a3*b3)/c);
}


inline void c_orthogonal_polar_factor(
            float a11, float a12, float a13,
            float a21, float a22, float a23,
            float a31, float a32, float a33,
            float &r11, float &r12, float &r13,
            float &r21, float &r22, float &r23,
            float &r31, float &r32, float &r33)
{
    float u11, u12, u13, u21, u22, u23, u31, u32, u33;
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;

    svd(
        a11, a12, a13, a21, a22, a23, a31, a32, a33,
        u11, u12, u13, u21, u22, u23, u31, u32, u33,
        r11, r12, r13, r21, r22, r23, r31, r32, r33,
        v11, v12, v13, v21, v22, v23, v31, v32, v33);

    multAB(
        u11, u12, u13, u21, u22, u23, u31, u32, u33,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
        r11, r12, r13, r21, r22, r23, r31, r32, r33);
}


inline void c_orthogonal_polar_factor_with_diagonal_sum(
            float a11, float a12, float a13,
            float a21, float a22, float a23,
            float a31, float a32, float a33,
            float &r11, float &r12, float &r13,
            float &r21, float &r22, float &r23,
            float &r31, float &r32, float &r33, float &s)
{
    float u11, u12, u13, u21, u22, u23, u31, u32, u33;
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;

    svd(
        a11, a12, a13, a21, a22, a23, a31, a32, a33,
        u11, u12, u13, u21, u22, u23, u31, u32, u33,
        r11, r12, r13, r21, r22, r23, r31, r32, r33,
        v11, v12, v13, v21, v22, v23, v31, v32, v33);

    s = r11 + r22 + r33;

    multAB(
        u11, u12, u13, u21, u22, u23, u31, u32, u33,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
        r11, r12, r13, r21, r22, r23, r31, r32, r33);
}
#endif


