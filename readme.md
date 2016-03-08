###SMMC
A programm written during a Mathematical Contest in Modeling in late 2015.
SMMC is a manifold clustering method solving the hybrid nonlinear manifold clustering problem, 
which is able to handle situations where the manifolds on which the data points lie are 
(a) linear and/or nonlinear and (b) intersecting and/or not intersecting. It first applies MPPCA 
(Mixtures of Probabilistic Principal Component Analyzers) to break the intersection regions into
a collection of good local patches according to the underlying clusters/maniflods which is an 
important step to separate multiple manifolds with intersections, then uses spectral clustering 
with a suitable constructed affinity matrix to find the clusters.

![Alt text](https://raw.githubusercontent.com/pyhong/Manifold-Learning/master/Pic/2c.png)  

![Alt text](https://raw.githubusercontent.com/pyhong/Manifold-Learning/master/Pic/2d.png)  

![Alt text](https://raw.githubusercontent.com/pyhong/Manifold-Learning/master/Pic/4a1_1.png)  

![Alt text](https://raw.githubusercontent.com/pyhong/Manifold-Learning/master/Pic/4a1_2.png)  

It's nice.


