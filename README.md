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
| orthogonal polar factor      | \|\|RTR - I\|\|\_F  |                   9.247e-08 | 5.602e-07 | 
| orthonormal procrutes problem| \|\|RX - Y\|\|\_F /  \|\|Y\|\|\_F|      6.240e-08 | 2.407e-07 | 
| orthogonal procrutes problem*| \|\|RSX - Y\|\|\_F  /  \|\|Y\|\|\_F |   5.555e-06 | 9.115e-06 | 

## execution time
|        method                |   evaluation        | numpy with openblas | ours              |
| ---------------------------- | --------------------|--------------------:|------------------:|
| orthogonal polar factor      | average             | 57.629us            | 2.435us           |
| orthonormal procrutes problem| average             | 62.557us            | 4.945us           |
| orthogonal procrutes problem*| average             | 2238.998us          | 89.131us          |

(*) the orthogonal procrutes problem uses iterative method to approximate the solution, iteration number is set to 30.

## execution time/reconstruction error tradeoff of orthogonal procrutes problem
![alt text](https://github.com/derleeG/pyprocrutes/blob/master/fig/Figure.png "Tradeoff plot")
