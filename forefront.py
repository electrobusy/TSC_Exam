# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:10:10 2021

@author: Rohan
"""

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.linalg import block_diag

import time

#%% Functions

def forward(t,s,u,param):
    """ 
    Simulated dynamics of a space debris model
    
    **inputs:**
        - *t*: time
        - *s*: state
        - *u*: control inputs
        - *param*: model parameters
    **outputs:** 
        - *ds*: derivative of the final state 
    """
    
    m = param['m'] # mass [kg]
    S = param['S'] # reference surface [m^2]
    d_ext = param['d_ext'] # exterior diameter [m]
    Iyy = param['Iyy']
    
    C_D = param['C_D'] # drag coefficient
    C_L = param['C_L'] # lift coefficient
    C_M = param['C_M'] # moment coefficient
    
    A_C = np.array([[0, 1],[0, 0]])
    A = block_diag(A_C,A_C,A_C)
    
    B_CD = np.array([0,1])*(S/m)*C_D
    B_CL = np.array([0,1])*(S/m)*C_L 
    B_CM = np.array([0,1])*(S*d_ext/Iyy)*C_M 
    
    B = u*np.concatenate((B_CD,B_CL,B_CM))
    # NOTE: u is the dynamic pressure at each datapoint (here considered as an 
    # input of the system)
    
    ds = A.dot(s) + B
    
    return ds

def propagate_traj(t,x0,u,param):
    """ 
    Propagate trajectory of dynamical model via numerical integration
    
    **inputs:**
        - *t*: time
        - *x0*: initial state
        - *u*: control inputs
        - *param*: model parameters
    **outputs:** 
        - *x*: state path
    """
    
    # store solution
    x = np.zeros((len(x0),len(t))) # state of the system
    
    # initial condition
    x[:,0] = x0
    
    # propagate 
    for i in range(len(t)-1):
        # time-step
        ts = [t[i],t[i+1]]
        ts = np.array(ts)
        
        s0 = x[:,i]
        
        # ode - get next state (for the next time-step)
        y = odeint(forward, y0=s0, t=ts, args=(u[i],param), tfirst=True)
        
        # save solution
        x[:,i+1] = y[-1,:].T
    
    return x 

def cost(X,t,y,u,param):
    """ 
    Cost function used in the inverse method: 
        L2-norm of the trajectory error
    
    **inputs:**
        - *X*: parameters to be estimated
        - *t*: time
        - *y*: original response
        - *u*: control inputs
        - *param*: model parameters
    **outputs:** 
        - *x*: state path
    """
    
    # print('-- Solve for $C_D$ = {:f} / $C_L$ = {:f} / $C_M$ = {:f}'.format(X[0], X[1], X[2]))
    
    # update new k from the optimizer
    param['C_D'] = X[0]
    param['C_L'] = X[1]
    param['C_M'] = X[2]
    
    # start timer - for ode
    # start_time = time.time()
    
    # ode - get next state
    s0 = y[:,0]
    x = propagate_traj(t, s0, u, param)
        
    # end time - for ode
    # end_time = time.time() - start_time 
    
    # print('ODE end time: %f'%end_time)
    
    # cost --> original position response minus the obtained response with the 
    # (parameter being optimized)
    dev_x = x[0,:] - y[0,:]
    dev_z = x[2,:] - y[2,:]
    dev_theta = x[4,:] - y[4,:]
    
    # L2 norm of the position error
    err_x = np.linalg.norm(dev_x, ord=2) 
    err_z = np.linalg.norm(dev_z, ord=2)  
    err_theta = np.linalg.norm(dev_theta, ord=2) 
    
    err = np.sqrt(err_x**2 + err_z**2 + err_theta**2)

    # print('Error = %f'%err)

    # return value 
    return err     
        
#%% Main script
if __name__ == "__main__":
    
    # -- close figures
    plt.close('all')
    
    # rad to deg
    rad2deg = 180/np.pi
    
    # -- initial state
    s0 = np.array([0, 0, 0, 0, 0, 0]) 

    # -- parameters 
    param = dict()
    # C_D
    param['C_D'] = 2.36 # [-]
    # C_L
    param['C_L'] = 1.15 # [-]
    # C_M
    param['C_M'] = 2e-3 # [-]
    
    #  mass
    param['m'] = 46.047e-3 # [kg]
    # lenghts and reference surface
    d_ext = 60.11e-3 # [m]
    h = 15e-3 # [m]
    # Moments of inertia
    Iyy = 20.360e-6 # [kg.m^2]
    
    # Store parameters
    param['S'] = d_ext*h # [m^2]
    param['d_ext'] = d_ext # [m]
    param['Iyy'] = Iyy # [kg.m^2]
    
    # -- time and input vectors (for the response)
    # time
    T = 10e-3 # [sec] -> longshot run time
    nframes = 50 
    t = np.linspace(0, T, num=nframes, endpoint=True)
    # external input --> dynamic pressure (it has an exponential decay)
    # NOTE: The following values indicate the coefficients of fit for this decay
    A1 = 11.1221
    B1 = -0.0959
    C1 = 0.0015
    u = np.exp(A1 + B1*t + C1*t**2)
    
    # fig = plt.figure()
    # plt.plot(t,u)
    # plt.ylabel(r'$q_\infty$ [-]')
    # plt.xlabel(r'$t$ [sec]')
    # plt.grid()
    
    # -- generate response
    s_raw = propagate_traj(t, s0, u, param)
    
    # -- add noise to the trajectory
    s = s_raw + 0.001*np.random.randn(len(s0),len(t))
    
    # plot reponses
    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(t, s_raw[0,:], color='r', label='original (C_D=%f)'%param['C_D'])
    ax[0,0].scatter(t, s[0,:], s=10, label='noisy', marker='^')
    ax[0,0].set_xlabel(r'$t$ [sec]')
    ax[0,0].set_ylabel(r'$x$ [m]')
    ax[0,0].grid()
    
    ax[0,1].plot(t,s_raw[2,:], color='r', label='original (C_L=%f)'%param['C_L'])
    ax[0,1].scatter(t, s[2,:], s=10, label='noisy', marker='^')
    ax[0,1].set_xlabel(r'$t$ [sec]')
    ax[0,1].set_ylabel(r'$z$ [m]')
    ax[0,1].grid()
    
    ax[0,2].plot(t,s_raw[4,:]*rad2deg, color='r', label='original (C_M=%f)'%param['C_M'])
    ax[0,2].scatter(t, s[4,:]*rad2deg, s=10, label='noisy', marker='^')
    ax[0,2].set_xlabel(r'$t$ [sec]')
    ax[0,2].set_ylabel(r'$\theta$ [deg]')
    ax[0,2].grid()
    
    #%% -- Inverse method (decifer the value of C_D/C_L/C_M from the response) 
    
    # initial condition
    X = [0.5,0.5,0.5] # initial condition for [C_D,C_L,C_M] 
    # bounds for coefficients (during the search)
    from scipy.optimize import Bounds
    bounds = Bounds(-5, 5, True)
    # optimization
    optimizer = 'Nelder-Mead'
    res = minimize(cost, X, args=(t,s,u,param), method=optimizer, 
                    bounds=bounds, options={'ftol': 1e-6, 'disp': True})
    
    # -- get parameters (from the optimization) 
    CD_pred = res.x[0]
    CL_pred = res.x[1]
    CM_pred = res.x[2]
    
    # -- generate response with k from the optimization
    param['C_D'] = CD_pred
    param['C_L'] = CL_pred
    param['C_M'] = CM_pred
    s0_pred = s[:,0]
    s_pred = propagate_traj(t, s0_pred, u, param)
    
    # -- superimpose response
    ax[0,0].scatter(t, s_pred[0,:], label='predicted (C_D={:f}))'.format(CD_pred), 
                    s=10, color='orange')
    ax[0,1].scatter(t, s_pred[2,:], label='predicted (C_L={:f})'.format(CL_pred), 
                    s=10, color='orange')
    ax[0,2].scatter(t, s_pred[4,:]*rad2deg, label='predicted (C_M={:f})'.format(CM_pred),
                    s=10, color='orange')
    
    ax[0,0].legend()
    ax[0,1].legend()
    ax[0,2].legend()
    
    # -- absolute error of the responses (and plot it)
    dev = s_pred - s
    
    ax[1,0].scatter(t, dev[0,:], s=10, color='orange')
    ax[1,0].set_xlabel(r'$t$ [sec]')
    ax[1,0].set_ylabel(r'$e_x$ [m]')
    ax[1,0].grid()
    
    ax[1,1].scatter(t, dev[2,:], s=10, color='orange')
    ax[1,1].set_xlabel(r'$t$ [sec]')
    ax[1,1].set_ylabel(r'$e_z$ [m]')
    ax[1,1].grid()
    
    ax[1,2].scatter(t, dev[4,:], s=10, color='orange')
    ax[1,2].set_xlabel(r'$t$ [sec]')
    ax[1,2].set_ylabel(r'$e_\theta$ [deg]')
    ax[1,2].grid()
    
    fig.suptitle(r'Inverse method: 3-DoF Dynamic problem -> Initial condition $C_D$ = {:f} / $C_L$ = {:f} / $C_M$ = {:f})'.format(X[0], X[1], X[2]))
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    #%% Inverse method (estimate C_D/C_L/C_M) + Bootstrapping (for unc. quantification)
    # NOTE: USE PARALLEL PROCESSING TO SPEED UP THIS CODE
    
    from sklearn.model_selection import train_test_split
    
    # number of trials
    n_trials = 50
    
    # initialize array of coefficient to save result for each trial
    C_D_arr = np.zeros(n_trials)  
    C_L_arr = np.zeros(n_trials)  
    C_M_arr = np.zeros(n_trials)  
    
    # percentage of training data
    p_train = 0.7
    
    # initial condition
    X = [0.5, 0.5, 0.5] # initial condition for [C_D,C_L,C_M] 
    
    # optimizer
    optimizer = 'Nelder-Mead'
    
    # bounds for coefficients (during the search)
    from scipy.optimize import Bounds
    bounds = Bounds(-5, 5, True) # [-5,5] --> for all of them (change later)
    
    # main loop
    start_time = time.time()
    for i in range(n_trials):
        
        # split data (train/test)
        # print('Split data with percentages (train/test) %f/%f'%(p_train,1-p_train))
        t_train, t_test = train_test_split(t, train_size=p_train)
        
        # NOTE: the previous function splits the data in a unsorter manner. 
        # One needs to sort the data in taking into account the time. 
        idx_train = t_train.argsort()
        idx_test = t_test.argsort()
        
        # -- sort data
        # time
        t_train = t_train[idx_train] 
        t_test = t_test[idx_test]
        # states
        s_train = s[:,idx_train]
        s_test = s[:,idx_test]
        # "inputs" - corresponds to dynamic pressure (in this case)
        u_train = u[idx_train]
        u_test = u[idx_test]
        
        ####  Optimization
        print('-------------------------------')
        print('Trial %d - %s Method' %(i+1,optimizer))
        optimization_time = time.time()
        res = minimize(cost, X, args=(t_train,s_train,u_train,param), method=optimizer, 
                    bounds=bounds, options={'ftol': 1e-6, 'disp': True})
        # Print results and total elapsed time
        print('Result: $C_D$ = %f / C_L = %f / C_M = %f' %(res.x[0],res.x[1],res.x[2]))
        print('Optimization time: %f' %(time.time() - optimization_time))
        
        # store coefficients
        C_D_arr[i] = res.x[0]
        C_L_arr[i] = res.x[1]
        C_M_arr[i] = res.x[2]
        
    print('TOTAL elapsed time: %f' %(time.time() - start_time))
    
    # save coefficients (to process later)
    # import pickle 
    # import os 
    
    # filename = open(os.path.join(os.getcwd(),'coef.p'), 'wb') # file
    # pickle.dump([C_D_arr, C_L_arr, C_M_arr], filename) # pickle the file
    
    #%% Statistics for each coefficients
    
    C_label = ['C_D', 'C_L', 'C_M']
    C_arr = [C_D_arr, C_L_arr, C_M_arr]
    
    C_mean = list()
    C_std = list()
    
    for (arr,label) in zip(C_arr,C_label):
        # mean 
        C_mean.append(np.mean(arr))
        # std 
        C_std.append(np.std(arr))
        
        print('%s = %f +/- %f'%(label,C_mean[-1],C_std[-1]*1.96))
    
    #%% 
    # histogram of for each coefficient
    fig, axs = plt.subplots(3,1)
    
    for (ax,arr,label) in zip(axs,C_arr,C_label): 
        ax.hist(arr, bins='auto')
        ax.set_xlabel(r'$' + label + '$ [-]')
        ax.set_ylabel('# of counts')
        ax.grid()
    
    
    
    
    
    
    
    
    