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

#endif


