import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from hpl_functions import plot_final, plot_closed_loop, plotFromFile, vx_interp, ey_interp, vehicle_model
from Track_new import *
from matplotlib.patches import Polygon
from hplStrategy import hplStrategy
from hplStrategy import ExactGPModel
from hplControl_LB import hplControl
from matplotlib.cm import ScalarMappable

map = Map2(0.55, 'LShape')
map.plot_map()
# MPC parameters
T = 5 # number of environment prediction samples --> training input will have length 21
ds = 1.5 # environment sample step (acts on s)
N_mpc = 2 # number of timesteps (of length dt) for the MPC horizon
dt = 0.1 # MPC sample time (acts on t). determines how far out we take our training label and env. forecast. (N*dt)
fint = 0.5 # s counter which determines start of new training data
model = 'BARC' # vehicle model, BARC or Genesis

thread1 = None
# safety-mpc parameters
gamma = 0.1 # cost function weight (tracking vt vs. centerline)
vt = 1 # speed for lane-keeping safety controller

# flag for retraining
retrain_flag = False

# flag for plotting sets
plotting_flag = False

# flag for whether to incorporate safety constraint by measuring accepted risk level (1 = use safety, 0 = ignore safety)
beta = False

# AeBeUsStrat = hplStrategy(T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, retrain_flag)
HPLMPC = hplControl(T, N_mpc, dt, map, vt)

x_state = np.array([0, 0, 0, 0]) 
u_state = np.array([0,0])
counter = 0

x_closed_loop = np.zeros_like(x_state)
x_closed_loop = x_state
X = []
Y = []

def stop_plot():
    time.sleep(0.2)
    plt.close()

while counter < 200:
    map.plot_map()
    X.append(x_state[0])
    Y.append(x_state[1])
    plt.scatter(X, Y)
    thread1 = threading.Thread(target=stop_plot)
    thread1.start()
    plt.show()
    counter += 1
    x_pred, u_pred, status = HPLMPC.solve(x_state, u_state)
    # print(l)
    # u_state = l[1][:][0]
    # u_state = 
    # print(u_state)
    u_state = u_pred[:,0]

    x_state = vehicle_model(x_state, u_state, dt, map, model)
    print(f'X: {x_state[0]}')
    print(f'Y: {x_state[1]}')
    print(f'Accel: {u_state[0]}')
    print(f'Delta: {u_state[1]}')
    print(f'Velocity: {x_state[2]}')
    
    # x_closed_loop = np.vstack(x_closed_loop, x_state)


# plt.plot(x_closed_loop[0], x_closed_loop[1])

plt.show()
