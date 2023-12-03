#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the paper describing the control framework:
#
#     [1] Charlott Vallon and Francesco Borrelli. "Data-driven hierarchical predictive learning in unknown
#         environments." In IEEE CASE (2020).
#
#
# Attibution Information: Code developed by Charlott Vallon
# (for clarifiactions and suggestions please write to charlottvallon@berkeley.edu).
#
# ----------------------------------------------------------------------------------------------------------------------
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from hpl_functions import plot_final, plot_closed_loop, plotFromFile, vx_interp, ey_interp, vehicle_model
from Track_new import *
from matplotlib.patches import Polygon
from hplStrategy import hplStrategy
from hplStrategy import ExactGPModel
from hplControl import hplControl


#%% Initialization 

from matplotlib.cm import ScalarMappable

# define the test environment from loaded pickle file, and load previously determined raceline
# testfile = 'Tracks/JP.pkl'
# [raceline_X,raceline_Y, Cur, map, race_time, pframe] = plotFromFile(testfile, lineflag=True)
# -0.542275
# map = Map2(1.1, 'LShape')
map = Map2(0.55, 'LShape')
# map.plot_map()
# MPC parameters
T = 15 # number of environment prediction samples --> training input will have length 21
ds = 1.5 # environment sample step (acts on s)
N_mpc = 20 # number of timesteps (of length dt) for the MPC horizon
dt = 0.1 # MPC sample time (acts on t). determines how far out we take our training label and env. forecast. (N*dt)
fint = 0.5 # s counter which determines start of new training data
model = 'BARC' # vehicle model, BARC or Genesis

# hpl parameters
s_conf_thresh = 8 # confidence threshold for del_s prediction
ey_conf_thresh = 1.3 # confidence threshold for ey prediction
thread1 = None
# safety-mpc parameters
gamma = 0.5 # cost function weight (tracking vt vs. centerline)
vt = 1 # speed for lane-keeping safety controller

# flag for retraining
retrain_flag = False

# flag for plotting sets
plotting_flag = False

# flag for whether to incorporate safety constraint by measuring accepted risk level (1 = use safety, 0 = ignore safety)
beta = False

AeBeUsStrat = hplStrategy(T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, retrain_flag)
HPLMPC = hplControl(T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, vt, model, gamma, beta)


#%% Control Loop

# initialize the vehicle state
# vx0 = vx_interp(pframe, 0)
# print(f'VX0: {vx0}')
# ey0 = ey_interp(pframe,0)
# print(f'EY0: {ey0}')
# 7.5754822039753895
# EY0: 0.5252233678397118
x_state = np.array([vt*0.8, 0, 0, 0, 0, 0.3]) # vx, vy, wz, epsi, s, ey
# x_state = np.array([5, 0, 0, 0, 0, 0.5252233678397118]) # vx, vy, wz, epsi, s, ey
# x_state = np.array([vx0, 0, 0, 0, 0, ey0]) # vx, vy, wz, epsi, s, ey

# initialize vectors to save vehicle state and inputs
x_closedloop = np.reshape(x_state, (6, 1))
u_closedloop = np.array([[0],[0]])
x_pred = x_closedloop
# x_pred_stored = np.empty((1,21))
x_pred_stored = np.empty((1,21))
u_pred = np.array([[0,0],[0,0]])

counter = 0
# while the predicted s-state of the vehicle is less than track_length:

def stop_plot():
    time.sleep(0.3)
    plt.close()


print(f'MAP Tracklength: {map.TrackLength}')
# while x_pred[4, -1] < map.TrackLength:
while counter <= 250:
    counter +=1
    # evaluate GPs
    est_s, std_s, est_ey, std_ey, strategy_set, centers = AeBeUsStrat.evaluateStrategy(x_state)

    #print('Centers: ')
    #print(centers)
    # evaluate control
    x_pred, u_pred = HPLMPC.solve(x_state, std_s, std_ey, centers)
    
    # store predicted state signals
    try:
        x_pred_stored = np.vstack((x_pred_stored, x_pred))
    except:
        pass
    
    # append applied input
    u = u_pred[:,0]
    u_closedloop = np.hstack((u_closedloop, np.reshape(u,(2,1))))
    
    # apply input to system 
    x_state = vehicle_model(x_state, u, dt, map, model)
    # round to avoid numerical disasters
    eps = 1e-4
    x_state = np.array([round(i,3) for i in x_state])
    while abs(x_state[2])>=1.569:
        x_state[2] -= eps*sign(x_state[2])
    while abs(x_state[5])>=0.8:
        x_state[5] -= eps*sign(x_state[5])
    
    # save important quantities for next round (predicted inputs, predicted state)
    try:
        x_closedloop = np.hstack((x_closedloop, np.reshape(x_state,(6,1))))
    except Exception as e:
        pass

    # plot the closedloop thus far 
    # fig, ax = plt.subplots(1)
    # ax.add_patch(strategy_set)
    print(counter)
    #print(x_pred[2])
    # print(x_pred[0])
    # print(x_pred[1])


    if plotting_flag:
        for st in HPLMPC.set_list:
            if st != []:
                rect_pts_xy = np.array([map.getGlobalPosition(st[0] - st[1], st[2] - st[3],0)])
                rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(map.getGlobalPosition(st[0] + st[1], st[2] - st[3], 0), (1,-1))))
                rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(map.getGlobalPosition(st[0] + st[1], st[2] + st[3], 0), (1,-1))))
                rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(map.getGlobalPosition(st[0] - st[1], st[2] + st[3], 0), (1,-1))))
                ax.add_patch(Polygon(rect_pts_xy, True, color = 'g',alpha = 0.3))
       
    if counter > 60:
        print(x_pred[4,-1])
        plot_closed_loop(map, x_closedloop, x_pred=x_pred[:, :HPLMPC.N + 1], offst=20)
        plt.show()
    #thread1 = threading.Thread(target=stop_plot)
    #thread1.start()

    #thread1.join()
    # plt.close()
x_closedloop = np.hstack((x_closedloop, x_pred))
hpl_time = np.shape(x_closedloop)[1]*dt
x_pred_stored = x_pred_stored[1:,:]
print(np.shape(x_closedloop))


for i in range(len(x_pred_stored[:, 0])):
    if i % 6 == 0:
        print('Vx at step ' + str(i/6) + ': ' + str(x_pred_stored[i,0]))
        print('Vy at step: ' + str(i / 6) + ': ' + str(x_pred_stored[i + 1, 0]))


#%% Plotting closed-loop behavior

# plot_closed_loop(map,x_closedloop,offst=1000)

# plt.plot(x_closedloop[4,:], ey_interp(pframe, x_closedloop[4,:]),'b',label='Raceline')
# plt.legend()
# plt.xlabel('s')
# plt.ylabel('e_y')
# plt.show()
#
plot_final(map,x_closedloop,offst=1000)

# plt.figure()
# plt.plot(x_closedloop[4,:], x_closedloop[0,:],'r',label='HPL')
# # plt.plot(x_closedloop[4,:], vx_interp(pframe, x_closedloop[4,:]),'b',label='Raceline')
# plt.legend()
# plt.xlabel('s')
# plt.ylabel('v_x')
plt.show()
