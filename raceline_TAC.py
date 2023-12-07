#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:41:34 2020

@author: vallon2
"""

import numpy as np
from pyomo.environ import *
from pyomo.dae import *
import sys
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time
#https://f1-circuits.netlify.com/
#https://github.com/jbelien/F1-Circuits/tree/master/public/data
sys.path.append('fnc')

################## PROTOCOL FOR FINDING NEW RACING LINES ###########################

#1. Use matlab script track_f1race.m to load a particular track (see Matlab file for different track options on the github page)
#2. matlab script will return txfilebreak and txfile. Add to Track_new.py as self.points and self.coefs, respectively.
#3. The last declared track in Track_new.py __init__ will be imported in this script.

vehicle_type = 'BARC' # options = BARC, Genesis, Indy
save_data_flag = True # should we save raceline data?
track_name = 'LShape'
filename_to_save = 'LShape.pkl'

# set vehicle parameters
if vehicle_type == 'BARC':
    mass  = 1.98
    lf = 0.125
    lr = 0.125
    Iz = 0.024
    Cf = 1.25
    Cr = 1.25
    Df = 0.8 * mass * 9.81 / 2.0
    Bf = 1.0
    Dr = 0.8 * mass * 9.81 / 2.0
    Br = 1.0
elif vehicle_type == 'Genesis':
    mass  = 2303.1
    lf = 1.5213
    lr = 1.4987
    Iz = 5520.1
    Cr = 13.4851e4*2 # what are these -__-
    Cf = 7.6419e4*2
elif vehicle_type == 'Indy':
    mass = 630.21 # kg
    lf = 1.48 # mapproximate - these are a range
    lr = 1.48  #m
    Iz = 550 # from simulator
    Cr = 246037 # N/rad
    Cf = 178164 # N/rad
else:
    print('Specify correct vehicle type')

###################################
# plotting functions

def plotter(subplot, x, *series, **kwds):
    plt.subplot(subplot)
    for i,y in enumerate(series):
        plt.plot(x, [value(y[t]) for t in x], 'brgcmk'[i%6]+kwds.get('points',''))
    plt.title(kwds.get('title',''))
    plt.legend(tuple(y.cname() for y in series))
    plt.xlabel(x.cname())

def plotTrajectory(map, X,Y):
    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        print(map.getGlobalPosition(i * 0.1, 0))
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)
    plt.figure()
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(X, Y, '-r')

def plotTrajectory_newmap(map, X,Y):
    map.plot_map()
    plt.plot(X, Y, '-r')
    
def plotTrajectoryFromDF(map,DF):
    X=[0]
    Y=[DF['x5'][0]]
    for j in range(1,len(DF)):      
        cur_s = DF.index[j]
        ey=DF['x5'][cur_s]
        tmp = map.getGlobalPosition(cur_s, ey, 0)
        X.append(tmp[0])
        Y.append(tmp[1])
    print(X)
    plotTrajectory_newmap(map, X,Y)
    
def getValues(DF, s):
    m1x4 = np.interp(s, DF.index, DF['x4'].values)
    m1x5 = np.interp(s, DF.index, DF['x5'].values)
    
    return m1x4, m1x5


####################################

def find_raceline(map, vehicle_type, obj_num):
    
    #SYSTEM STATES:  vx=x[0],  vy=x[1], wz=x[2] ,e_psi=x[3], t=x[4], e_y=x[5]
    #SYSTEM INPUTS:  ax[m/s^2]=u0, steering(rad)=u1
    #INDEPENDENT VARIABLE IS s (space)
    
    model = m = ConcreteModel()
    m.sf = Param(initialize = TrackLength)
    m.s = ContinuousSet(bounds=(0, m.sf))
    if vehicle_type == 'BARC':
        m.u0 = Var(m.s, bounds=(-1,1), initialize=0) 
        m.u1 = Var(m.s, bounds=(-0.5,0.5), initialize=0) 
        m.alpha_f = Var(m.s, initialize=0)
        m.alpha_r = Var(m.s, initialize=0)
        m.Fyf = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.s,bounds=(0,10), initialize=0.01) 
        m.x1 = Var(m.s, bounds=(-10,10),initialize=0) 
        m.x2 = Var(m.s,bounds=(-0.5*3.14,0.5*3.14), initialize=0)
        m.x3 = Var(m.s, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.s, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.s, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    elif vehicle_type == 'Genesis':
        m.u0 = Var(m.s, bounds=(-4,4), initialize=0) # 4m/s2, internet
        m.u1 = Var(m.s, bounds=(-0.5,0.5), initialize=0) # will keep the same, but unclear
        m.alpha_f = Var(m.s, initialize=0)
        m.alpha_r = Var(m.s, initialize=0)
        m.Fyf = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.s,bounds=(0,67), initialize=0.01) #150mph, according to internet
        m.x1 = Var(m.s, initialize=0)
        m.x2 = Var(m.s, initialize=0)
        m.x3 = Var(m.s, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.s, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.s, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    elif vehicle_type == 'Indy':
        m.u0 = Var(m.s, bounds=(-8.3,8.3), initialize=0) # internet
        m.u1 = Var(m.s, bounds=(-0.43,0.43), initialize=0) # internet
        m.alpha_f = Var(m.s, initialize=0)
        m.alpha_r = Var(m.s, initialize=0)
        m.Fyf = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.s,bounds=(0,102.8), initialize=0.01) #230mph, according to internet
        m.x1 = Var(m.s, initialize=0)
        m.x2 = Var(m.s, initialize=0)
        m.x3 = Var(m.s, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.s, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.s, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    else:
        'Specify correct vehicle type'
    m.dx0ds = DerivativeVar(m.x0, wrt=m.s)
    m.dx1ds = DerivativeVar(m.x1, wrt=m.s)
    m.dx2ds = DerivativeVar(m.x2, wrt=m.s)
    m.dx3ds = DerivativeVar(m.x3, wrt=m.s)
    m.dx4ds = DerivativeVar(m.x4, wrt=m.s)
    m.dx5ds = DerivativeVar(m.x5, wrt=m.s)
    m.du1ds = DerivativeVar(m.u1, wrt=m.s)
    # to avoid divide by 0
    eps=0.000001
    
    #Objective function
    if obj_num == 0:
        m.obj = Objective(expr=m.x4[m.sf], sense=minimize)
    elif obj_num == 1:
        m.obj = Objective(expr= m.x4[m.sf] + 0.1*sum(m.du1ds[i] for i in m.s), sense=minimize)
    elif obj_num == 2:
        m.obj = Objective(expr= m.x4[m.sf] + 0.01*sum(m.du1ds[i] for i in m.s), sense=minimize)
    elif obj_num == 3:
        m.obj = Objective(expr= m.x4[m.sf] + 0.001*sum(m.du1ds[i] for i in m.s), sense=minimize)
    elif obj_num == 4:
        m.obj = Objective(expr= m.x4[m.sf] + 0.005*sum(m.du1ds[i] for i in m.s), sense=minimize)
    
    #ways to tune the cost: 
        # penalize u1
        # penalize u2
        # penalize u1 and u2
        # vary the penalizations on u1 and u2
        # penalize the input rate
    # to do: write this optimization problem as a function that gets called
    # different arguments will choose a different objective tuning
    # function output should be the DF, then we can store them all
    
    #sideslip and lateral force
    def _alphafc(m, s):
        return m.alpha_f[s] == m.u1[s] - atan((m.x1[s] + lf * m.x2[s])/ (m.x0[s]))
    m.c4 = Constraint(m.s, rule=_alphafc)
    def _alpharc(m, s):
        return m.alpha_r[s] == -atan((m.x1[s] - lr * m.x2[s])/ (m.x0[s]))
    m.c3 = Constraint(m.s, rule=_alpharc)
    def _Fyfc(m, s):
        return m.Fyf[s] ==  Df * Cf * Bf * m.alpha_f[s]
    m.c2 = Constraint(m.s, rule=_Fyfc)
    def _Fyrc(m, s):
        return m.Fyr[s] ==  Dr * Cr * Br * m.alpha_r[s]
    m.c1 = Constraint(m.s, rule=_Fyrc)
    
    #Differential model definition
    def _x0dot(m, s):
        cur = map.getCurvature(s)
        print(cur)
        return m.dx0ds[s] == (m.u0[s] - 1 / mass *  m.Fyf[s] * sin(m.u1[s]) + m.x2[s]*m.x1[s])*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x0dot = Constraint(m.s, rule=_x0dot)
    
    def _x1dot(m, s):
        cur = map.getCurvature(s)
        return m.dx1ds[s] == (1 / mass * (m.Fyf[s] * cos(m.u1[s]) + m.Fyr[s]) - m.x2[s] * m.x0[s])*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x1dot = Constraint(m.s, rule=_x1dot)
    
    def _x2dot(m, s):
        cur = map.getCurvature(s)
        return m.dx2ds[s] == (1 / Iz *(lf*m.Fyf[s] * cos(m.u1[s]) - lr * m.Fyr[s]) )*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x2dot = Constraint(m.s, rule=_x2dot)
    
    def _x3dot(m, s):
        cur = map.getCurvature(s)
        return m.dx3ds[s] == ( m.x2[s]*(1 - cur * m.x5[s])/(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])) - cur)
    m.x3dot = Constraint(m.s, rule=_x3dot)
    
    def _x4dot(m, s):
        cur = map.getCurvature(s)
        return m.dx4ds[s] == ((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x4dot = Constraint(m.s, rule=_x4dot)
    
    def _x5dot(m, s):
        cur = map.getCurvature(s)
        return m.dx5ds[s] == (m.x0[s] * sin(m.x3[s]) + m.x1[s] * cos(m.x3[s]))*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x5dot = Constraint(m.s, rule=_x5dot)
    
    # min and max constraints on steering
    def _u1dotmax(m, s):
        cur = map.getCurvature(s)
        return m.du1ds[s] <= 0.5*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.i1dot = Constraint(m.s, rule=_u1dotmax)
    def _u1dotmin(m, s):
        cur = map.getCurvature(s)
        return m.du1ds[s] >= -0.5*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.i1dot = Constraint(m.s, rule=_u1dotmin)
    
    # inital and terminal conditions
    def _init(m):
        yield m.x5[0] == m.x5[TrackLength]
        yield m.x0[0] == m.x0[TrackLength]
        yield m.x1[0] == m.x1[TrackLength]
        yield m.x2[0] == m.x2[TrackLength]
        yield m.x3[0] == m.x3[TrackLength]
        yield m.x4[0] == 0
        yield m.x5[0] == m.x5[TrackLength]
    m.init_conditions = ConstraintList(rule=_init)
    
    # Discretize model using radau or finite difference collocation
    #TransformationFactory('dae.collocation').apply_to(m, nfe=50, ncp=5, scheme='LAGRANGE-LEGENDRE'  )
    #TransformationFactory('dae.collocation').apply_to(m, nfe=200, ncp=10, scheme='LAGRANGE-RADAU'  )
    #TransformationFactory('dae.collocation').apply_to(m, nfe=100, ncp=10, scheme='LAGRANGE-RADAU')
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=2200 )
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=3000 )
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=10000)
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=int(np.floor(map.TrackLength)*2))
    TransformationFactory('dae.collocation').apply_to(m, nfe=200, ncp=10, scheme='LAGRANGE-LEGENDRE') #STANDARD METHOD
    
    
    # Solve algebraic model
    solver = SolverFactory('ipopt')
    # Solver options
    solver.options['max_iter'] = 500 #8000
    #solver.options = {'tol': 1e-6,
    #                  'mu_init': 1e-8,
    #                  'bound_push': 1e-8,
    #                 'halt_on_ampl_error': 'yes'}
    results = solver.solve(m,tee=True)
    
    # Plot results
    plotter(141, m.s, m.x0, m.x1, title='Differential Variables')
    plotter(142, m.s, m.x2, m.x3, title='Differential Variables')
    plotter(143, m.s, m.x4, m.x5, title='Differential Variables', points='o')
    plotter(144, m.s, m.u0, m.u1,m.Fyf ,m.Fyr,  title='Control Variable', points='o')
    #plt.show()
    
    # build global position of vehicle
    Psi=value(m.x3[0])
    X=[0]
    Y=[value(m.x5[0])]
    Cur=[0]
    EY=[value(m.x5[0])]
    svec=list(sorted(m.s.value))
    for j in range(1,len(svec)):
        sj=svec[j]
        ey=value(m.x5[sj])
        tmp = map.getGlobalPosition(sj, ey, 0)
        cv = map.getCurvature(sj)
        EY.append(ey)
        Cur.append(cv)
        X.append(tmp[0])
        Y.append(tmp[1])
    plotTrajectory_newmap(map, X,Y)
    plt.show()
    
    DF = pd.DataFrame()
    for v in m.component_objects(Var,active=True):
        for index in v:
            DF.at[index, v.name] = value(v[index])   
    
    return DF


def find_second_raceline(map, vehicle_type, obj_num):
    
    #SYSTEM STATES:  vx=x[0],  vy=x[1], wz=x[2] ,e_psi=x[3], t=x[4], e_y=x[5]
    #SYSTEM INPUTS:  ax[m/s^2]=u0, steering(rad)=u1
    #INDEPENDENT VARIABLE IS s (space)
    
    model = m = ConcreteModel()
    m.sf = Param(initialize = TrackLength)
    m.s = ContinuousSet(bounds=(0, m.sf))
    if vehicle_type == 'BARC':
        m.u0 = Var(m.s, bounds=(-1,1), initialize=0) 
        m.u1 = Var(m.s, bounds=(-0.5,0.5), initialize=0) 
        m.alpha_f = Var(m.s, initialize=0)
        m.alpha_r = Var(m.s, initialize=0)
        m.Fyf = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.s,bounds=(0,10), initialize=0.01) 
        m.x1 = Var(m.s, bounds=(-10,10),initialize=0) 
        m.x2 = Var(m.s,bounds=(-0.5*3.14,0.5*3.14), initialize=0)
        m.x3 = Var(m.s, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.s, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.s, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    elif vehicle_type == 'Genesis':
        m.u0 = Var(m.s, bounds=(-4,4), initialize=0) # 4m/s2, internet
        m.u1 = Var(m.s, bounds=(-0.5,0.5), initialize=0) # will keep the same, but unclear
        m.alpha_f = Var(m.s, initialize=0)
        m.alpha_r = Var(m.s, initialize=0)
        m.Fyf = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.s,bounds=(0,67), initialize=0.01) #150mph, according to internet
        m.x1 = Var(m.s, initialize=0)
        m.x2 = Var(m.s, initialize=0)
        m.x3 = Var(m.s, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.s, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.s, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    elif vehicle_type == 'Indy':
        m.u0 = Var(m.s, bounds=(-8.3,8.3), initialize=0) # internet
        m.u1 = Var(m.s, bounds=(-0.43,0.43), initialize=0) # internet
        m.alpha_f = Var(m.s, initialize=0)
        m.alpha_r = Var(m.s, initialize=0)
        m.Fyf = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.s,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.s,bounds=(0,102.8), initialize=0.01) #230mph, according to internet
        m.x1 = Var(m.s, initialize=0)
        m.x2 = Var(m.s, initialize=0)
        m.x3 = Var(m.s, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.s, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.s, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    else:
        'Specify correct vehicle type'
    m.dx0ds = DerivativeVar(m.x0, wrt=m.s)
    m.dx1ds = DerivativeVar(m.x1, wrt=m.s)
    m.dx2ds = DerivativeVar(m.x2, wrt=m.s)
    m.dx3ds = DerivativeVar(m.x3, wrt=m.s)
    m.dx4ds = DerivativeVar(m.x4, wrt=m.s)
    m.dx5ds = DerivativeVar(m.x5, wrt=m.s)
    m.du1ds = DerivativeVar(m.u1, wrt=m.s)
    # to avoid divide by 0
    eps=0.000001
    
    #Objective function
    if obj_num == 0:
        m.obj = Objective(expr=m.x4[m.sf], sense=minimize)
    elif obj_num == 1:
        m.obj = Objective(expr= m.x4[m.sf] + 0.1*sum(m.du1ds[i] for i in m.s), sense=minimize)
    elif obj_num == 2:
        m.obj = Objective(expr= m.x4[m.sf] + 0.01*sum(m.du1ds[i] for i in m.s), sense=minimize)
    elif obj_num == 3:
        m.obj = Objective(expr= m.x4[m.sf] + 0.001*sum(m.du1ds[i] for i in m.s), sense=minimize)
    elif obj_num == 4:
        m.obj = Objective(expr= m.x4[m.sf] + 0.005*sum(m.du1ds[i] for i in m.s), sense=minimize)

    
    #sideslip and lateral force
    def _alphafc(m, s):
        return m.alpha_f[s] == m.u1[s] - atan((m.x1[s] + lf * m.x2[s])/ (m.x0[s]))
    m.c4 = Constraint(m.s, rule=_alphafc)
    def _alpharc(m, s):
        return m.alpha_r[s] == -atan((m.x1[s] - lr * m.x2[s])/ (m.x0[s]))
    m.c3 = Constraint(m.s, rule=_alpharc)
    def _Fyfc(m, s):
        return m.Fyf[s] ==  Df * Cf * Bf * m.alpha_f[s]
    m.c2 = Constraint(m.s, rule=_Fyfc)
    def _Fyrc(m, s):
        return m.Fyr[s] ==  Dr * Cr * Br * m.alpha_r[s]
    m.c1 = Constraint(m.s, rule=_Fyrc)
    
    #Differential model definition
    def _x0dot(m, s):
        cur = map.getCurvature(s)
        print(cur)
        return m.dx0ds[s] == (m.u0[s] - 1 / mass *  m.Fyf[s] * sin(m.u1[s]) + m.x2[s]*m.x1[s])*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x0dot = Constraint(m.s, rule=_x0dot)
    
    def _x1dot(m, s):
        cur = map.getCurvature(s)
        return m.dx1ds[s] == (1 / mass * (m.Fyf[s] * cos(m.u1[s]) + m.Fyr[s]) - m.x2[s] * m.x0[s])*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x1dot = Constraint(m.s, rule=_x1dot)
    
    def _x2dot(m, s):
        cur = map.getCurvature(s)
        return m.dx2ds[s] == (1 / Iz *(lf*m.Fyf[s] * cos(m.u1[s]) - lr * m.Fyr[s]) )*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x2dot = Constraint(m.s, rule=_x2dot)
    
    def _x3dot(m, s):
        cur = map.getCurvature(s)
        return m.dx3ds[s] == ( m.x2[s]*(1 - cur * m.x5[s])/(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])) - cur)
    m.x3dot = Constraint(m.s, rule=_x3dot)
    
    def _x4dot(m, s):
        cur = map.getCurvature(s)
        return m.dx4ds[s] == ((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x4dot = Constraint(m.s, rule=_x4dot)
    
    def _x5dot(m, s):
        cur = map.getCurvature(s)
        return m.dx5ds[s] == (m.x0[s] * sin(m.x3[s]) + m.x1[s] * cos(m.x3[s]))*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.x5dot = Constraint(m.s, rule=_x5dot)
    
    # min and max constraints on steering
    def _u1dotmax(m, s):
        cur = map.getCurvature(s)
        return m.du1ds[s] <= 0.5*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.i1dot = Constraint(m.s, rule=_u1dotmax)
    def _u1dotmin(m, s):
        cur = map.getCurvature(s)
        return m.du1ds[s] >= -0.5*((1 - cur * m.x5[s]) /(eps+m.x0[s] * cos(m.x3[s]) - m.x1[s] * sin(m.x3[s])))
    m.i1dot = Constraint(m.s, rule=_u1dotmin)
    
    # racing constraints: t2(s) + ey2(s) >= t1(s) + ey1(s) + buffer
    buffer = 2
    def _raceConstraint(m,s):
        m1x4, m1x5 = getValues(DF1, s)
        return m.x4[s] + m.x5[s] >= m1x4 + m1x5 + buffer
    
    
    # inital and terminal conditions
    def _init(m):
        yield m.x5[0] == m.x5[TrackLength]
        yield m.x0[0] == m.x0[TrackLength]
        yield m.x1[0] == m.x1[TrackLength]
        yield m.x2[0] == m.x2[TrackLength]
        yield m.x3[0] == m.x3[TrackLength]
        yield m.x4[0] == 0
        yield m.x5[0] == m.x5[TrackLength]
    m.init_conditions = ConstraintList(rule=_init)
    
    # Discretize model using radau or finite difference collocation
    #TransformationFactory('dae.collocation').apply_to(m, nfe=50, ncp=5, scheme='LAGRANGE-LEGENDRE'  )
    #TransformationFactory('dae.collocation').apply_to(m, nfe=200, ncp=10, scheme='LAGRANGE-RADAU'  )
    #TransformationFactory('dae.collocation').apply_to(m, nfe=100, ncp=10, scheme='LAGRANGE-RADAU')
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=2200 )
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=3000 )
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=10000)
    #TransformationFactory('dae.finite_difference').apply_to(m, nfe=int(np.floor(map.TrackLength)*2))
    TransformationFactory('dae.collocation').apply_to(m, nfe=200, ncp=10, scheme='LAGRANGE-LEGENDRE') #STANDARD METHOD
    
    
    # Solve algebraic model
    solver = SolverFactory('ipopt')
    # Solver options
    solver.options['max_iter'] = 300
    #solver.options = {'tol': 1e-6,
    #                  'mu_init': 1e-8,
    #                  'bound_push': 1e-8,
    #                 'halt_on_ampl_error': 'yes'}
    results = solver.solve(m,tee=True)
    
    # Plot results
    plotter(141, m.s, m.x0, m.x1, title='Differential Variables')
    plotter(142, m.s, m.x2, m.x3, title='Differential Variables')
    plotter(143, m.s, m.x4, m.x5, title='Differential Variables', points='o')
    plotter(144, m.s, m.u0, m.u1,m.Fyf ,m.Fyr,  title='Control Variable', points='o')
    #plt.show()
    
    # build global position of vehicle
    Psi=value(m.x3[0])
    X=[0]
    Y=[value(m.x5[0])]
    Cur=[0]
    EY=[value(m.x5[0])]
    svec=list(sorted(m.s.value))
    for j in range(1,len(svec)):
        sj=svec[j]
        ey=value(m.x5[sj])
        tmp = map.getGlobalPosition(sj, ey, 0)
        cv = map.getCurvature(sj)
        EY.append(ey)
        Cur.append(cv)
        X.append(tmp[0])
        Y.append(tmp[1])
    plotTrajectory_newmap(map, X,Y)
    plt.show()
    
    DF = pd.DataFrame()
    for v in m.component_objects(Var,active=True):
        for index in v:
            DF.at[index, v.name] = value(v[index])
    
    
    return DF
    
    
#################################### ACTUAL CODE ####################################

# from Track_new import Map2
# TrackHalfW=1.6/2
# track_list = ['CN_short', 'AT_short', 'MX_short', 'HU_short', 'CA_short', 'IT_short', 'JP_short']

# for track_name in track_list:
    
#     map = Map2(TrackHalfW, track_name)
#     TrackLength=map.TrackLength
#     plotTrajectory_newmap(map, [],[])
#     plt.show()
    
#     filename_to_save = track_name[0:3] + 'extra_tracks.pkl'
    
#     DF_1 = find_raceline(map, vehicle_type, 1)
#     DF_2 = find_raceline(map, vehicle_type, 2)
#     DF_3 = find_raceline(map, vehicle_type, 3)
#     DF_4 = find_raceline(map, vehicle_type, 4)
        
#     db = {} 
#     db['trackmap'] = map
#     db['DF1'] = DF_1
#     db['DF2'] = DF_2
#     db['DF3'] = DF_3
#     db['DF4'] = DF_4
    
#     if save_data_flag:         
#         pickle.dump(db, open( filename_to_save, "wb" ) )
        
        
from Track_new import Map2
track_name = 'LShape'
TrackHalfW=0.55
map = Map2(TrackHalfW, track_name)
TrackLength=map.TrackLength
start_time = time.time()
DF = find_raceline(map, vehicle_type, 0)
duration = time.time()-start_time
print(duration)
plotTrajectoryFromDF(map,DF)

def constraintCheck(DF1, DF2):
    # select the s index from one of them
    s_indices = DF1.index
    
    # time plus ey for DF1
    df1vals = DF1['x4'].values + DF1['x5'].values
    df2vals = DF2['x4'].values + DF2['x5'].values
    
    plt.figure()
    plt.plot(df2vals-df1vals)
    
####################### SAVING DATA ########################      
# save the data into a file, be careful to uncomment this only when we want to save new data
# what should we save? x0-x5, u0, u1, map
# if save_data_flag:
    
#     db = {} 
#     db['trackmap'] = map
#     db['DF1'] = DF
#     db['DF2'] = DF_2
#     db['DF3'] = DF3
#     db['DF4'] = DF4
             
#     pickle.dump(db, open( filename_to_save, "wb" ) )
    
###################### NEXT STEPS ###################
# Write architecture for training data collection (make a few training datas, and use those to write. Then get more data)

