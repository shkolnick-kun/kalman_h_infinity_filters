# Adaptive Kalman-H-infinity filter prototypes in python
This repo contains python prototypes for a family of adaptive filters.

Kalman filter is used as a base filtering algoritm and H-infinity filter as fallback algoritm in case of Kalman filter divergence.

We've used [filterpy](https://github.com/rlabbe/filterpy) library code for documentation and API design.
In fact [ExtendedKalmanFilter](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py) is the base class for two of four filters in this repo.

We've also used [Kalman and Bayesian Filters in Python book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) code for tests and plots.
