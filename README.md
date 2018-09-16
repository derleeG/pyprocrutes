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


# Benchmark 
testing method: average over 10000 randomly initialized matrices
## reconstruction error
|        method                |   evaluation        | numpy with openblas | ours              |  
| ---------------------------- | --------------------|--------------------:|------------------:|
| orthogonal polar factor      | \|\|RTR - I\|\|\_F  |                   9.229e-08 | 5.501e-07 | 
| orthonormal procrutes problem| \|\|RX - Y\|\|\_F /  \|\|Y\|\|\_F|      6.208e-08 | 1.250e-06 | 
| orthogonal procrutes problem*| \|\|RSX - Y\|\|\_F  /  \|\|Y\|\|\_F |   4.819e-06 | 1.044e-05 | 

## execution time
|        method                |   evaluation        | numpy with openblas | ours              |
| ---------------------------- | --------------------|--------------------:|------------------:|
| orthogonal polar factor      | average             | 57.089us            | 2.394us           |
| orthonormal procrutes problem| average             | 61.737us            | 4.933us           |
| orthogonal procrutes problem*| average             | 2169.875us          | 89.183us          |

(*) the orthogonal procrutes problem uses iterative method to approximate the solution, iteration number is set to 30.
