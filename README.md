# pyprocrutes
A python library for solving orthogonal Procrutes problem and orthonormal Procrutes problem.

# Problem formulation
we want to solve the following argmin problems. 
see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
and http://empslocal.ex.ac.uk/people/staff/reverson/uploads/Site/procrustes.pdf for details.
## Orthonormal Procrutes problem
R = argmin\_R \|\|RX - Y\|\|\_F, subject to RTR = I and det(R) = 1.
## Orthogonal Procrutes problem
R, S = argmin\_R, S \|\|RSX - Y\|\|\_F, subject to RTR = I , det(R) = 1 and S is a diagonal matrix with positive values.
