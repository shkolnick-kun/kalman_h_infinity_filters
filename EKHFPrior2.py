# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments
"""
Copyright 2019 Paul A Beltyukov
Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from copy import deepcopy
import numpy as np
from numpy import dot, eye
from scipy.stats import chi2
import scipy.linalg as linalg
from filterpy.kalman import ExtendedKalmanFilter

class ExtendedKalmanHinfFilterPrior2(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u=0, alpha=0.01, eps_mul=0.1):
        ExtendedKalmanFilter.__init__(self, dim_x, dim_z, dim_u)
        self.beta_n = chi2.ppf(1.0 - alpha, dim_z)
        self._eps_mul = eps_mul
        
    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman/Hinfinity 
        filter with prior residuals used for Hinfinity correction.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, posterior is not computed

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """
        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)
            
        #Will check for folter Divergence
        H = HJacobian(self.x, *args)
        hx = Hx(self.x, *hx_args)
        nu = residual(z, hx)

        PHT = self.P.dot(H.T)
        C = H.dot(PHT)
        self.S = C + R
        self.SI = linalg.inv(self.S)
        
        thr = self.beta_n
        if dot(nu.T, dot(self.SI, nu)) >= thr: 
            #Divergence detected, H-infinity correction needed
            nutnu = dot(nu.T, nu)
            k  = nutnu * nutnu / thr - dot(nu.T, dot(self.S, nu))
            k /= dot(nu.T, dot(C , nu))
            self.P *= 1.0 + k
            
            #Need to recompute PHT, self.S and self.SI due to self.P update
            PHT = dot(self.P, H.T)
            self.S = H.dot(PHT) + R
            self.SI = linalg.inv(self.S)
            

        #Now we may update self.K, self.P, self.y, self.x
        self.y = nu
        self.K = PHT.dot(self.SI)
        self.x += dot(self.K, nu)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)
        
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        """ Performs the predict/update innovation of the extended Kalman
        filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, along with the
           optional arguments in args, and returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx after the required state
            variable.

        u : np.array or scalar
            optional control vector input to the filter.
        """
        self.predict(u)
        self.update(z, HJacobian, Hx, self.R, args, hx_args, residual=np.subtract)    
