# TSC_Exam April 2021: Rohan Chotalal
=====================================

This documentation describes the FOREFRONT (FOR Estimation oF aeRodynamic cOefficieNTs) module. It was developed to estimate the aerodynamic coefficients using information of the trajectory of a space debris model at hypersonic speeds.  

The main application of this work is to predict trajectories with accuracy using such aerodynamic models.   

The current version of this code relies on the generation of a synthetic data representing the motion (position + attitude) of a dynamical system with constant aerodynamic coefficients. 

Folder structure:
----------------

This repository is organized as follows: 

TSC_Exam_2021
|_ forefront.py
|_ README.txt

The forefront.py script is the main code of this repository. All the specific functions required for the application of an inverse method for estimation of aerodynamic coefficients are detailed in this file. 

The code also includes the implementation of a bootstrapping method to determine the uncertainty bounds of this coefficient. Such method is still presented here on a preliminary basis. However, 

Currently, the code relies on data generated synthetically using 3-DoF model derived from Newton and Euler's laws. The next step relies on using real data from wind tunnel experiments.

Requirements:
-------------

To run the code, the following packages are required: 
	numpy
	matplotlib
	scipy
	sklearn

Detailed documentation is provided in [documentation/build/html](documentation/build/html)



	

