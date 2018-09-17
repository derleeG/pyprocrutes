# pyprocrutes
A python library for solving orthogonal Procrutes problem and several generalized Procrutes problem.

# Problem formulation
we want to solve the following argmin problems. 
see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
and http://empslocal.ex.ac.uk/people/staff/reverson/uploads/Site/procrustes.pdf for details.
## Orthogonal Procrutes problem
R = argmin\_R \|\|RX - Y\|\|\_F, subject to RTR = I and det(R) = 1.
## Isotropic Procrutes problem
R, s = argmin\_R, s \|\|RsX - Y\|\|\_F, subject to RTR = I , det(R) = 1 and S is a positive scalar.
## Anisotropic Procrutes problem
R, S = argmin\_R, S \|\|RSX - Y\|\|\_F, subject to RTR = I , det(R) = 1 and S is a diagonal matrix with positive values.


# Benchmark 
testing method: average over 10000 randomly initialized matrices
## reconstruction error
|        method                |   evaluation        | numpy with openblas | ours              |  
| ---------------------------- | --------------------|--------------------:|------------------:|
| orthogonal polar factor      | \|\|RTR - I\|\|\_F  |                   1.422e-07 | 5.517e-07 | 
| orthogonal procrutes problem | \|\|X - RTY\|\|\_F /  \|\|X\|\|\_F|      7.548e-08 | 2.376e-07 | 
| Isotropic procrutes problem  | \|\|X - RTY/s\|\|\_F  /  \|\|X\|\|\_F |   1.144e-07 | 2.149e-07 | 
| Anisotropic procrutes problem*| \|\|X - S^-1RTY\|\|\_F  /  \|\|X\|\|\_F |   1.345e-05 | 1.369e-05 | 

## execution time
|        method                |   evaluation        | numpy with openblas | ours              |
| ---------------------------- | --------------------|--------------------:|------------------:|
| orthogonal polar factor      | average             | 59.589us            | 2.585us           |
| orthogonal procrutes problem | average             | 62.854us            | 5.002us           |
| Isotropic procrutes problem  | average             | 80.728us            | 6.379us           |
| Anisotropic procrutes problem*| average            | 2276.941us          | 89.834us          |

(*) For anisotropic procrutes problem, solution is approximated with an iterative method. Iteration number is set to 30.

## execution time/reconstruction error tradeoff of orthogonal procrutes problem
![alt text](https://github.com/derleeG/pyprocrutes/blob/master/fig/Figure.png "Tradeoff plot")
