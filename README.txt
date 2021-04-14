# TSC_Exam April 2021: Rohan Chotalal
=====================================

This documentation describes the FOREFRONT (FOR Estimation oF aeRodynamic cOefficieNTs) module. It was developed to compute the aerodynamic coefficients using information of the motion of a space debris model at hypersonic speeds.  

The main application of this work is correctly identify aerodynamic models of space debris  

The current version of this code relies on the generation of a synthetic data representing the motion (position + attitude) of a dynamical system with constant aerodynamic coefficients. 

Folder structure:
----------------

This repository is organized as follows: 

TSC_Exam_2021
|_ forefront.py
|_ README.txt

The ethiopian.py script is the main code of this repository. All the specific functions required for the application of an inverse method for estimation of aerodynamic coefficients are detailed in this file. It also provides implementation of a bootstrapping method to determine the uncertainty bounds of this coefficient. 

Currently, the code relies on data generated synthetically using 3-DoF model derived from Newton and Euler's laws. The next step on using real data. 

Requirements:
-------------

To run the code, the following packages are required: 
	numpy
	matplotlib
	scipy
	sklearn

Detailed documentation is provided in [documentation/build/html](documentation/build/html)


	

