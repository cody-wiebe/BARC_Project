#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
# from pyomo.environ import *
# from pyomo.dae import *
import casadi as ca
from casadi import MX, sum1, vertcat, atan2, asin, acos, sin, cos


class hplControl():
    def __init__(self, T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, vt, model, gamma, beta):

        self.T = T # number of environment prediction samples --> training input will have length 21
        self.ds = ds # environment sample step (acts on s)
        self.N_mpc = N_mpc # number of timesteps (of length dt) for the MPC horizon 
        self.dt = dt # MPC sample time (acts on t). determines how far out we take our training label and env. forecast. (N*dt)
        self.fint = fint # s counter which determines start of new training data
        self.s_conf_thresh = s_conf_thresh # confidence threshold for del_s prediction
        self.ey_conf_thresh = ey_conf_thresh # confidence threshold for ey prediction
        self.map = map
        self.model = model
        self.vt = vt
        self.gamma = gamma
        self.beta = beta
        
        # initialize control horizon to N_mpc
        self.N = N_mpc
        
        self.set_list = []
        for i in range(self.N_mpc):
            self.set_list.append([])
                   
        # scaler info is used to transform the states into standardized ones
        self.scaler = pickle.load(open('SafetyControl/scaler.pkl','rb'))
        self.clf = pickle.load(open('SafetyControl/clf.pkl','rb'))

        self.init_time = True
        self.first_time = True
        self.prev_N = -1

        self.TrackHalfW = None
        self.TrackLength = None
        self.opti = None
        self.mass = None
        self.lf = None
        self.lr = None
        self.Iz = None
        self.Df = None
        self.Cf = None
        self.Bf = None
        self.Dr = None
        self.Cr = None
        self.Br = None
        self.sf = None
        self.u0 = None
        self.u1 = None
        self.alpha_f = None
        self.alpha_r = None
        self.Fyf = None
        self.Fyr = None
        self.x0 = None
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.x4 = None
        self.x5 = None
        self.slack0 = None
        self.slack1 = None
        self.slack2 = None
        self.slack3 = None
        self.slack4 = None
        self.slack5 = None

    def init_vars(self, x, N, accel):
        self.init_time = False
        self.prev_N = N

        self.TrackHalfW = self.map.width
        self.TrackLength = self.map.TrackLength
        self.opti = ca.Opti()
        self.mass = 1.98
        self.lf = 0.125
        self.lr = 0.125
        self.Iz = 0.024
        self.Df = 0.8 * self.mass * 9.81 / 2.0
        self.Cf = 1.25
        self.Bf = 1.0
        self.Dr = 0.8 * self.mass * 9.81 / 2.0
        self.Cr = 1.25
        self.Br = 1.0
        self.sf = self.opti.parameter()
        self.opti.set_value(self.sf, self.TrackLength)
        self.u0 = self.opti.variable(N - 1)
        self.u1 = self.opti.variable(N - 1)
        self.alpha_f = self.opti.variable(N)
        self.alpha_r = self.opti.variable(N)
        self.Fyf = self.opti.variable(N)
        self.Fyr = self.opti.variable(N)
        self.x0 = self.opti.variable(N)
        self.x1 = self.opti.variable(N)
        self.x2 = self.opti.variable(N)
        self.x3 = self.opti.variable(N)
        self.x4 = self.opti.variable(N)
        self.x5 = self.opti.variable(N)
        self.slack0 = self.opti.variable()
        self.slack1 = self.opti.variable()
        self.slack2 = self.opti.variable()
        self.slack3 = self.opti.variable()
        self.slack4 = self.opti.variable()
        self.slack5 = self.opti.variable()

        self.opti.subject_to(self.opti.bounded(-1, self.u0, 1))
        self.opti.subject_to(self.opti.bounded(-0.5, self.u1, 0.5))
        self.opti.subject_to(self.opti.bounded(-2, self.alpha_f, 2))
        self.opti.subject_to(self.opti.bounded(-2, self.alpha_r, 2))
        self.opti.subject_to(self.opti.bounded(-self.mass * 9.8, self.Fyf, self.mass * 9.8))
        self.opti.subject_to(self.opti.bounded(-self.mass * 9.8, self.Fyr, self.mass * 9.8))
        self.opti.subject_to(self.opti.bounded(0, self.x0, 10))
        self.opti.subject_to(self.opti.bounded(-10, self.x1, 10))
        self.opti.subject_to(self.opti.bounded(-0.5 * 3.14, self.x2, 0.5 * 3.14))
        self.opti.subject_to(self.opti.bounded(-0.3 * 3.1416, self.x3, 0.3 * 3.1416))
        self.opti.subject_to(self.opti.bounded(0, self.x4, self.sf))
        self.opti.subject_to(self.opti.bounded(-self.TrackHalfW, self.x5, self.TrackHalfW))
        self.opti.set_initial(self.u0, 0)
        self.opti.set_initial(self.u1, 0)
        self.opti.set_initial(self.alpha_f, 0)
        self.opti.set_initial(self.alpha_r, 0)
        self.opti.set_initial(self.Fyf, 0)
        self.opti.set_initial(self.Fyr, 0)
        self.opti.set_initial(self.x0, 0)
        self.opti.set_initial(self.x1, 0.01)
        self.opti.set_initial(self.x2, 0)
        self.opti.set_initial(self.x4, 0)
        self.opti.set_initial(self.x5, 0)
        self.opti.set_initial(self.slack0, 0)
        self.opti.set_initial(self.slack1, 0)
        self.opti.set_initial(self.slack2, 0)
        self.opti.set_initial(self.slack3, 0)
        self.opti.set_initial(self.slack4, 0)
        self.opti.set_initial(self.slack5, 0)

        if accel:
            sum = 0
            for i in range(N):
                sum += (self.vt - self.x0[i]) ** 2
            self.opti.minimize(1000 * (self.slack2 ** 2 + self.slack3 ** 2) + sum)
        else:
            sum = 0
            for i in range(N):
                sum += (self.x5[i] - 0.2) ** 2 + self.gamma * (self.vt - self.x0[i]) ** 2
            self.opti.minimize(1000 * (self.slack2 ** 2 + self.slack3 ** 2) + sum)

        solver_opts = {'ipopt': {'print_level': 0}}
        self.opti.solver("ipopt", solver_opts)

    def curv_pred_LMPC(self, x, N):
   
        pred_curve = self.map.getCurvature(x[4])
        # current vx value
        v = x[0]
        
        # what s-values will we have if we have constant velocity (vt) for the next N steps?
        # assume that the velocity is applied directly along s 
        for i in range(1,N+1):
            pred_curve = np.hstack((pred_curve, self.map.getCurvature(x[4]+i*v*self.dt)))
            if v >= self.vt:
                v = max(v-1, self.vt)
            else:
                v = min(v+1, self.vt)
        
        return pred_curve
    
    def lane_track_MPC(self, x, N):        
        
        # calculates the center-lane tracking MPC control while keeping the velocity close to vd
        # gamma tunes weight of tracking centerline vs tracking velocity (suggest 0.1)
        cv_pred = self.curv_pred_LMPC(x, N)            
        TrackHalfW = self.map.width
        TrackLength = self.map.TrackLength
        # opti = ca.Opti()

        if self.model == 'BARC':
            mass  = 1.98
            lf = 0.125
            lr = 0.125
            Iz = 0.024
            Df = 0.8 * mass * 9.81 / 2.0
            Cf = 1.25
            Bf = 1.0
            Dr = 0.8 * mass * 9.81 / 2.0
            Cr = 1.25
            Br = 1.0   
            model = m = ConcreteModel()
            m.sf = Param(initialize = TrackLength)
            m.t = RangeSet(0, N)
            m.tnotN = m.t - [ N ]
            m.u0 = Var(m.tnotN, bounds=(-1,1), initialize=0)
            m.u1 = Var(m.tnotN, bounds=(-0.5,0.5), initialize=0)
            m.alpha_f = Var(m.t, bounds=(-2,2), initialize=0)
            m.alpha_r = Var(m.t, bounds=(-2,2), initialize=0)
            m.Fyf = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.Fyr = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.x0 = Var(m.t,bounds=(0,10), initialize=0.01) #vx - we can only move forward
            m.x1 = Var(m.t,bounds=(-10,10), initialize=0) #vy
            m.x2 = Var(m.t,bounds=(-0.5*3.14,0.5*3.14), initialize=0) #wz
            m.x3 = Var(m.t, bounds=(-0.3*3.1416,0.3*3.1416)) #epsi
            m.x4 = Var(m.t, bounds=(0,m.sf), initialize=0) #s
            m.x5 = Var(m.t, bounds=(-TrackHalfW,TrackHalfW), initialize=0) #ey
            m.slack0 = Var(initialize=0)
            m.slack1 = Var(initialize=0)
            m.slack2 = Var(initialize=0)
            m.slack3 = Var(initialize=0)
            m.slack4 = Var(initialize=0)
            m.slack5 = Var(initialize=0)
        
        if self.model == 'Genesis':
            mass  = 2303.1
            lf = 1.5213
            lr = 1.4987
            Iz = 5520.1
            Cr = 13.4851e4*2
            Cf = 7.6419e4*2
            
            Df = 0.8 * mass * 9.81 / 2.0
            Dr = 0.8 * mass * 9.81 / 2.0
            Br = 1.0   
            Bf = 1.0
            
            model = m = ConcreteModel()
            m.sf = Param(initialize = TrackLength)
            m.t = RangeSet(0, N) 
            m.tnotN = m.t - [ N ]
            m.u0 = Var(m.tnotN, bounds=(-4,4), initialize=0) 
            m.u1 = Var(m.tnotN, bounds=(-0.5,0.5), initialize=0)
            m.alpha_f = Var(m.t, initialize=0)
            m.alpha_r = Var(m.t, initialize=0)
            m.Fyf = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.Fyr = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.x0 = Var(m.t,bounds=(0,67), initialize=0.01) #150mph
            m.x1 = Var(m.t, initialize=0)
            m.x2 = Var(m.t, initialize=0)
            m.x3 = Var(m.t, bounds=(-0.3*3.1416,0.3*3.1416))
            m.x4 = Var(m.t, bounds=(0,20000), initialize=0)
            m.x5 = Var(m.t, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
        
        #Objective function -  minimize norm of ey along all predicted time steps, and deviation from vt      
        m.slack_obj = 1000*(m.slack2**2 + m.slack3**2)
        m.track_obj = sum(m.x5[i]**2 + self.gamma*(self.vt - m.x0[i])**2 for i in m.t)
        m.obj = Objective(expr = m.slack_obj + m.track_obj, sense=minimize)                        
        
        #sideslip and lateral force
        def _alphafc(m, t):
            return m.alpha_f[t] == m.u1[t] - atan((m.x1[t] + lf * m.x2[t])/ (m.x0[t]))
        m.c4 = Constraint(m.tnotN, rule=_alphafc)
        def _alpharc(m, t):
            return m.alpha_r[t] == -atan((m.x1[t] - lr * m.x2[t])/ (m.x0[t]))
        m.c3 = Constraint(m.tnotN, rule=_alpharc)
        def _Fyfc(m, t):
            return m.Fyf[t] ==  2 * Df * sin( Cf * atan(Bf * m.alpha_f[t]) )
        m.c2 = Constraint(m.tnotN, rule=_Fyfc)
        def _Fyrc(m, t):
            return m.Fyr[t] ==  2 * Dr * sin( Cr * atan(Br * m.alpha_r[t]) )
        m.c1 = Constraint(m.tnotN, rule=_Fyrc)
            
        def _x0(m,t): #vx
            return m.x0[t+1] == m.x0[t] + self.dt * (m.u0[t] - 1 / mass * m.Fyf[t] * sin(m.u1[t]) + m.x2[t]*m.x1[t])
        m.x0constraint = Constraint(m.tnotN, rule=_x0)    
        def _x1(m,t): #vy
            return m.x1[t+1] == m.x1[t] + self.dt * (1 / mass * (m.Fyf[t] * cos(m.u1[t]) + m.Fyr[t]) - m.x2[t]*m.x0[t])
        m.x1constraint = Constraint(m.tnotN, rule =_x1)    
        def _x2(m,t): # wz
            return m.x2[t+1] == m.x2[t] + self.dt * (1 / Iz * (lf * m.Fyf[t] * cos(m.u1[t]) - lr * m.Fyr[t]))
        m.x2constraint = Constraint(m.tnotN, rule=_x2)    
        def _x3(m,t): #epsi
            cur = cv_pred[t]
            return m.x3[t+1] == m.x3[t] + self.dt * (m.x2[t] - (m.x0[t] * cos(m.x3[t]) - m.x1[t]*sin(m.x3[t])) / (1 - cur*m.x5[t]) * cur)
        m.x3constraint = Constraint(m.tnotN, rule=_x3)    
        def _x4(m,t): #s
            cur = cv_pred[t]
            return m.x4[t+1] == m.x4[t] + self.dt * ( (m.x0[t] * cos(m.x3[t]) - m.x1[t]*sin(m.x3[t])) / (1 - cur * m.x5[t]) )
        m.x4constraint = Constraint(m.tnotN, rule=_x4)    
        def _x5(m,t): #ey
            return m.x5[t+1] == m.x5[t] + self.dt * (m.x0[t] * sin(m.x3[t]) + m.x1[t]*cos(m.x3[t]))
        m.x5constraint = Constraint(m.tnotN, rule=_x5)      
    
        def _init(m):
            yield m.x0[0] == x[0]
            yield m.x1[0] == x[1]
            yield m.x2[0] + m.slack2 == x[2] 
            yield m.x3[0] + m.slack3 == x[3] 
            yield m.x4[0] == x[4] 
            yield m.x5[0] == x[5] 
        m.init_conditions = ConstraintList(rule=_init)
        
        solver = SolverFactory('ipopt')
        solver.options["print_level"] = 1
        results = solver.solve(m,tee=False)   
        
        if results.solver.status == SolverStatus.ok:
            solver_flag = True
        else:
            solver_flag = False
            return 0, 0, solver_flag
        
        VX = value(m.x0[0])
        VY = value(m.x1[0])
        WZ = value(m.x2[0])
        EPSI = value(m.x3[0])
        S = value(m.x4[0])
        EY = value(m.x5[0])
        
        for t in range(1,N+1):
            VX = np.hstack((VX,(value(m.x0[t]))))
            VY = np.hstack((VY,(value(m.x1[t]))))
            WZ = np.hstack((WZ,(value(m.x2[t]))))
            EPSI = np.hstack((EPSI,(value(m.x3[t]))))
            S = np.hstack((S,(value(m.x4[t]))))
            EY = np.hstack((EY,(value(m.x5[t]))))
            
        x_pred = np.vstack((VX, VY, WZ, EPSI, S, EY))
        
        A = value(m.u0[0])
        DELTA = value(m.u1[0])
        
        for t in range(1,N):
            A = np.hstack((A, (value(m.u0[t]))))
            DELTA = np.hstack((DELTA, (value(m.u1[t]))))
     
        u_pred = np.vstack((A, DELTA))
        
        return [x_pred, u_pred, solver_flag]

    def lane_track_MPC_casadi(self, x, N):
        print("OG ME AYAAAAA")

        # calculates the center-lane tracking MPC control while keeping the velocity close to vd
        # gamma tunes weight of tracking centerline vs tracking velocity (suggest 0.1)
        cv_pred = self.curv_pred_LMPC(x, N)
        if self.first_time or self.prev_N != N:
            self.init_vars(x, N, False)

        self.first_time = False
        self.opti.subject_to()

        for i in range(N-1):
            cur = cv_pred[i]
            self.opti.subject_to(self.alpha_f[i] == self.u1[i] - atan2((self.x1[i] + self.lf * self.x2[i]), self.x0[i]))
            self.opti.subject_to(self.alpha_r[i] == -atan2((self.x1[i] - self.lr * self.x2[i]), self.x0[i]))
            self.opti.subject_to(self.Fyf[i] == 2 * self.Df * sin(self.Cf * atan2((self.Bf * self.alpha_f[i]), 1)))
            self.opti.subject_to(self.Fyr[i] == 2 * self.Dr * sin(self.Cr * atan2((self.Br * self.alpha_r[i]), 1)))
            self.opti.subject_to(self.x0[i + 1] == self.x0[i] + self.dt * (self.u0[i] - 1 / self.mass * self.Fyf[i] * sin(self.u1[i]) + self.x2[i] * self.x1[i]))
            self.opti.subject_to(self.x1[i + 1] == self.x1[i] + self.dt * (1 / self.mass * (self.Fyf[i] * cos(self.u1[i]) + self.Fyr[i]) - self.x2[i] * self.x0[i]))
            self.opti.subject_to(self.x2[i + 1] == self.x2[i] + self.dt * (1 / self.Iz * (self.lf * self.Fyf[i] * cos(self.u1[i]) - self.lr * self.Fyr[i])))
            self.opti.subject_to(self.x3[i + 1] == self.x3[i] + self.dt * (self.x2[i] - (self.x0[i] * cos(self.x3[i]) - self.x1[i] * sin(self.x3[i])) / (1 - cur * self.x5[i]) * cur))
            self.opti.subject_to(self.x4[i + 1] == self.x4[i] + self.dt * ((self.x0[i] * cos(self.x3[i]) - self.x1[i] * sin(self.x3[i])) / (1 - cur * self.x5[i])))
            self.opti.subject_to(self.x5[i + 1] == self.x5[i] + self.dt * (self.x0[i] * sin(self.x3[i]) + self.x1[i] * cos(self.x3[i])))

        self.opti.subject_to(self.x0[0] == x[0])
        self.opti.subject_to(self.x1[0] == x[1])
        self.opti.subject_to(self.x2[0] + self.slack2 == x[2])
        self.opti.subject_to(self.x3[0] + self.slack3 == x[3])
        self.opti.subject_to(self.x4[0] == x[4])
        self.opti.subject_to(self.x5[0] == x[5])

        sol = self.opti.solve()
        if sol.stats()['return_status'] == 'Solve_Succeeded':
            solver_flag = True
        else:
            solver_flag = False
            return 0, 0, solver_flag

        try:

            VX = sol.value(self.x0[0])
            VY = sol.value(self.x1[0])
            WZ = sol.value(self.x2[0])
            EPSI = sol.value(self.x3[0])
            S = sol.value(self.x4[0])
            EY = sol.value(self.x5[0])

            for t in range(1, N):

                VX = np.hstack((VX, (sol.value(self.x0[t]))))
                VY = np.hstack((VY, (sol.value(self.x1[t]))))
                WZ = np.hstack((WZ, (sol.value(self.x2[t]))))
                EPSI = np.hstack((EPSI, (sol.value(self.x3[t]))))
                S = np.hstack((S, (sol.value(self.x4[t]))))
                EY = np.hstack((EY, (sol.value(self.x5[t]))))

            x_pred = np.vstack((VX, VY, WZ, EPSI, S, EY))

            A = sol.value(self.u0[0])
            DELTA = sol.value(self.u1[0])

            for t in range(1, N-1):
                A = np.hstack((A, (sol.value(self.u0[t]))))
                DELTA = np.hstack((DELTA, (sol.value(self.u1[t]))))
            u_pred = np.vstack((A, DELTA))
            return [x_pred, u_pred, solver_flag]
        except Exception as e:
            print(e)

    def lane_track_MPC_accel(self, x, N):
        print("START WLE ME HAIIIIII")

        # calculates the center-lane tracking MPC control while keeping the velocity close to vd
        # gamma tunes weight of tracking centerline vs tracking velocity (suggest 0.1)
        cv_pred = self.curv_pred_LMPC(x, N)
        if self.init_time or self.prev_N != N:
            self.init_vars(x, N, True)

        self.opti.subject_to()

        for i in range(N - 1):
            cur = cv_pred[i]
            self.opti.subject_to(self.alpha_f[i] == self.u1[i] - atan2((self.x1[i] + self.lf * self.x2[i]), self.x0[i]))
            self.opti.subject_to(self.alpha_r[i] == -atan2((self.x1[i] - self.lr * self.x2[i]), self.x0[i]))
            self.opti.subject_to(self.Fyf[i] == 2 * self.Df * sin(self.Cf * atan2((self.Bf * self.alpha_f[i]), 1)))
            self.opti.subject_to(self.Fyr[i] == 2 * self.Dr * sin(self.Cr * atan2((self.Br * self.alpha_r[i]), 1)))
            self.opti.subject_to(self.x0[i + 1] == self.x0[i] + self.dt * (self.u0[i] - 1 / self.mass * self.Fyf[i] * sin(self.u1[i]) + self.x2[i] * self.x1[i]))
            self.opti.subject_to(self.x1[i + 1] == self.x1[i] + self.dt * (1 / self.mass * (self.Fyf[i] * cos(self.u1[i]) + self.Fyr[i]) - self.x2[i] * self.x0[i]))
            self.opti.subject_to(self.x2[i + 1] == self.x2[i] + self.dt * (1 / self.Iz * (self.lf * self.Fyf[i] * cos(self.u1[i]) - self.lr * self.Fyr[i])))
            self.opti.subject_to(self.x3[i + 1] == self.x3[i] + self.dt * (
                        self.x2[i] - (self.x0[i] * cos(self.x3[i]) - self.x1[i] * sin(self.x3[i])) / (1 - cur * self.x5[i]) * cur))
            self.opti.subject_to(
                self.x4[i + 1] == self.x4[i] + self.dt * ((self.x0[i] * cos(self.x3[i]) - self.x1[i] * sin(self.x3[i])) / (1 - cur * self.x5[i])))
            self.opti.subject_to(self.x5[i + 1] == self.x5[i] + self.dt * (self.x0[i] * sin(self.x3[i]) + self.x1[i] * cos(self.x3[i])))

        self.opti.subject_to(self.x0[0] == x[0])
        self.opti.subject_to(self.x1[0] == x[1])
        self.opti.subject_to(self.x2[0] + self.slack2 == x[2])
        self.opti.subject_to(self.x3[0] + self.slack3 == x[3])
        self.opti.subject_to(self.x4[0] == x[4])
        self.opti.subject_to(self.x5[0] == x[5])

        sol = self.opti.solve()

        if sol.stats()['return_status'] == 'Solve_Succeeded':
            solver_flag = True
        else:
            solver_flag = False
            return 0, 0, solver_flag

        try:

            VX = sol.value(self.x0[0])
            VY = sol.value(self.x1[0])
            WZ = sol.value(self.x2[0])
            EPSI = sol.value(self.x3[0])
            S = sol.value(self.x4[0])
            EY = sol.value(self.x5[0])

            for t in range(1, N):
                VX = np.hstack((VX, (sol.value(self.x0[t]))))
                VY = np.hstack((VY, (sol.value(self.x1[t]))))
                WZ = np.hstack((WZ, (sol.value(self.x2[t]))))
                EPSI = np.hstack((EPSI, (sol.value(self.x3[t]))))
                S = np.hstack((S, (sol.value(self.x4[t]))))
                EY = np.hstack((EY, (sol.value(self.x5[t]))))

            x_pred = np.vstack((VX, VY, WZ, EPSI, S, EY))

            A = sol.value(self.u0[0])
            DELTA = sol.value(self.u1[0])

            for t in range(1, N - 1):
                A = np.hstack((A, (sol.value(self.u0[t]))))
                DELTA = np.hstack((DELTA, (sol.value(self.u1[t]))))
            u_pred = np.vstack((A, DELTA))
            return [x_pred, u_pred, solver_flag]
        except Exception as e:
            print(e)
    def strategy_MPC(self, x, N):
        
        scaler = self.scaler
        clf = self.clf
        
        cv_pred = self.curv_pred_LMPC(x, N) 
    
        # centers contains [s_pred, s_std, ey_pred, ey_std]
        s_pred, s_std, ey_pred, ey_std = self.set_list[N-1]
        # print(s_pred)
        TrackHalfW = self.map.width
        #print(TrackHalfW)
        TrackLength = self.map.TrackLength
        
        # define model parameters
        if self.model == 'BARC':
            mass  = 1.98
            lf = 0.125
            lr = 0.125
            Iz = 0.024
            Df = 0.8 * mass * 9.81 / 2.0
            Cf = 1.25
            Bf = 1.0
            Dr = 0.8 * mass * 9.81 / 2.0
            Cr = 1.25
            Br = 1.0   
            model = m = ConcreteModel()
            m.sf = Param(initialize = TrackLength)
            m.t = RangeSet(0, N) 
            m.tnotN = m.t - [ N ]
            m.t_u = m.tnotN - [0]
            m.u0 = Var(m.tnotN, bounds=(-1,1), initialize=0)
            m.u1 = Var(m.tnotN, bounds=(-0.5,0.5), initialize=0)
            m.alpha_f = Var(m.t, bounds=(-2,2), initialize=0)
            m.alpha_r = Var(m.t, bounds=(-2,2), initialize=0)
            m.Fyf = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.Fyr = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.x0 = Var(m.t,bounds=(0,10), initialize=0.01) #vx - we can only move forward
            m.x1 = Var(m.t,bounds=(-10,10), initialize=0) #vy
            m.x2 = Var(m.t,bounds=(-0.5*3.14,0.5*3.14), initialize=0) #wz
            m.x3 = Var(m.t, bounds=(-0.3*3.1416,0.3*3.1416)) #epsi
            m.x4 = Var(m.t, bounds=(0,m.sf), initialize=0) #s
            m.x5 = Var(m.t, bounds=(-TrackHalfW,TrackHalfW), initialize=0) #ey
        
        elif self.model == 'Genesis':
            mass  = 2303.1
            lf = 1.5213
            lr = 1.4987
            Iz = 5520.1
            Cr = 13.4851e4*2
            Cf = 7.6419e4*2
            
            Df = 0.8 * mass * 9.81 / 2.0
            Dr = 0.8 * mass * 9.81 / 2.0
            Br = 1.0   
            Bf = 1.0
            
            model = m = ConcreteModel()
            m.sf = Param(initialize = TrackLength)
            m.t = RangeSet(0, N) 
            m.tnotN = m.t - [ N ]
            m.u0 = Var(m.tnotN, bounds=(-4,4), initialize=0) 
            m.u1 = Var(m.tnotN, bounds=(-0.5,0.5), initialize=0) 
            m.alpha_f = Var(m.t, initialize=0)
            m.alpha_r = Var(m.t, initialize=0)
            m.Fyf = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.Fyr = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
            m.x0 = Var(m.t,bounds=(0,67), initialize=0.01) #150mph
            m.x1 = Var(m.t, initialize=0)
            m.x2 = Var(m.t, initialize=0)
            m.x3 = Var(m.t, bounds=(-0.3*3.1416,0.3*3.1416))
            m.x4 = Var(m.t, bounds=(0,20000), initialize=0)
            m.x5 = Var(m.t, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
        
           
        #sideslip and lateral force
        def _alphafc(m, t):
            return m.alpha_f[t] == m.u1[t] - atan((m.x1[t] + lf * m.x2[t])/ (m.x0[t]))
        m.c4 = Constraint(m.tnotN, rule=_alphafc)
        def _alpharc(m, t):
            return m.alpha_r[t] == -atan((m.x1[t] - lr * m.x2[t])/ (m.x0[t]))
        m.c3 = Constraint(m.tnotN, rule=_alpharc)
        def _Fyfc(m, t):
            return m.Fyf[t] ==  2 * Df * sin( Cf * atan(Bf * m.alpha_f[t]) )
        m.c2 = Constraint(m.tnotN, rule=_Fyfc)
        def _Fyrc(m, t):
            return m.Fyr[t] ==  2 * Dr * sin( Cr * atan(Br * m.alpha_r[t]) )
        m.c1 = Constraint(m.tnotN, rule=_Fyrc)
            
        def _x0(m,t): #vx
            return m.x0[t+1] == m.x0[t] + self.dt * (m.u0[t] - 1 / mass * m.Fyf[t] * sin(m.u1[t]) + m.x2[t]*m.x1[t])
        m.x0constraint = Constraint(m.tnotN, rule=_x0)    
        def _x1(m,t): #vy
            return m.x1[t+1] == m.x1[t] + self.dt * (1 / mass * (m.Fyf[t] * cos(m.u1[t]) + m.Fyr[t]) - m.x2[t]*m.x0[t])
        m.x1constraint = Constraint(m.tnotN, rule =_x1)    
        def _x2(m,t): # wz
            return m.x2[t+1] == m.x2[t] + self.dt * (1 / Iz * (lf * m.Fyf[t] * cos(m.u1[t]) - lr * m.Fyr[t]))
        m.x2constraint = Constraint(m.tnotN, rule=_x2)    
        def _x3(m,t): #epsi
            cur = cv_pred[t]
            return m.x3[t+1] == m.x3[t] + self.dt * (m.x2[t] - (m.x0[t] * cos(m.x3[t]) - m.x1[t]*sin(m.x3[t])) / (1 - cur*m.x5[t]) * cur)
        m.x3constraint = Constraint(m.tnotN, rule=_x3)    
        def _x4(m,t): #s
            cur = cv_pred[t]
            return m.x4[t+1] == m.x4[t] + self.dt * ( (m.x0[t] * cos(m.x3[t]) - m.x1[t]*sin(m.x3[t])) / (1 - cur * m.x5[t]) )
        m.x4constraint = Constraint(m.tnotN, rule=_x4)    
        def _x5(m,t): #ey
            return m.x5[t+1] == m.x5[t] + self.dt * (m.x0[t] * sin(m.x3[t]) + m.x1[t]*cos(m.x3[t]))
        m.x5constraint = Constraint(m.tnotN, rule=_x5)        
    
        def _init(m):
            yield m.x0[0] == x[0]
            yield m.x1[0] == x[1]
            yield m.x2[0] == x[2] 
            yield m.x3[0] == x[3]
            yield m.x4[0] == x[4]
            yield m.x5[0] == x[5]
        m.init_conditions = ConstraintList(rule=_init)
        
        # terminal constraint - strategy
        def _termSet(m):
            yield m.x4[N] <= s_pred[0] + s_std[0]
            yield m.x4[N] >= s_pred[0] - s_std[0]
            yield m.x5[N] <= ey_pred[0] + ey_std[0]
            yield m.x5[N] >= ey_pred[0] - ey_std[0]
        m.term_constraint = ConstraintList(rule=_termSet)

        def _add_term(m):
            yield m.x0[N] <= 0.1
            # yield m.x5[N] <= 0.25
            # yield m.x1[N] == 0
            # yield m.x2[N] == 0

        if s_pred >= 17:
            m.term_constraint2 = ConstraintList(rule=_add_term)


        def _safetySet(m):
            w = clf.coef_[0]
            a = -w
            yield (a[0] * scaler.scale_[0] * (m.x0[N] - scaler.mean_[0])/scaler.var_[0]) + (a[1] * scaler.scale_[1] * (m.x1[N] - scaler.mean_[1])/scaler.var_[1]) + (a[2] * scaler.scale_[2] * (m.x2[N] - scaler.mean_[2])/scaler.var_[2]) + (a[3] * scaler.scale_[3] * (m.x3[N] - scaler.mean_[3])/scaler.var_[3]) + (a[4] * scaler.scale_[4] * (m.x5[N] - scaler.mean_[4])/scaler.var_[4]) - clf.intercept_[0] <= 0
        
        if self.beta:
            m.safety_constraint = ConstraintList(rule=_safetySet)
               
        # Objective function - should minimize deviation from state to center of corresponding strategy set
        v_set = 2
        index_list = [f for f,b in enumerate(self.set_list) if f < N and not np.array_equal(b, np.empty([0,]))]
        m.track_obj = sum(5*(self.set_list[i][0] - m.x4[i+1])**2 + 1.5*(self.set_list[i][2] - m.x5[i+1])**2 for i in index_list)
        diff_s = np.array([])
        diff_ey = np.array([])
        for i in index_list:
            diff_s = np.append(diff_s, self.set_list[i][0] - m.x4[i+1].value)
            diff_ey = np.append(diff_ey, self.set_list[i][2] - m.x5[i+1].value)
        m.inf_norm = max(diff_s)
        # m.track_obj = sum((self.set_list[i][0] - m.x4[i+1])**2 + (self.set_list[i][2] - m.x5[i + 1]) ** 2 for i in index_list)
        m.obj = Objective(expr =  m.track_obj + m.inf_norm, sense=minimize)
               
        solver = SolverFactory('ipopt')
        solver.options["print_level"] = 1
           
        try:
            results = solver.solve(m,tee=False)
            if results.solver.status == SolverStatus.ok:
                solver_flag = True
            else:
                solver_flag = False
                cv_pred
                return 0, 0, solver_flag
        except:
            print('weird solver problem')
            solver_flag = False
            cv_pred
            return 0, 0, solver_flag
        
        VX = value(m.x0[0])
        VY = value(m.x1[0])
        WZ = value(m.x2[0])
        EPSI = value(m.x3[0])
        S = value(m.x4[0])
        EY = value(m.x5[0])
        
        for t in range(1,N+1):
            VX = np.hstack((VX,(value(m.x0[t]))))
            VY = np.hstack((VY,(value(m.x1[t]))))
            WZ = np.hstack((WZ,(value(m.x2[t]))))
            EPSI = np.hstack((EPSI,(value(m.x3[t]))))
            S = np.hstack((S,(value(m.x4[t]))))
            EY = np.hstack((EY,(value(m.x5[t]))))
            
        x_pred = np.vstack((VX, VY, WZ, EPSI, S, EY))
        
        A = value(m.u0[0])
        DELTA = value(m.u1[0])
        
        for t in range(1,N):
            A = np.hstack((A, (value(m.u0[t]))))
            DELTA = np.hstack((DELTA, (value(m.u1[t]))))
     
        u_pred = np.vstack((A, DELTA))
        
        return x_pred, u_pred, solver_flag
        
    
    def solve(self, x_state, std_s, std_ey, centers):
        # depending on confidence measure, append the SetList
        # add the strategy set to the end of the set_list
        self.set_list.append(centers)

        # remove the first set from set_list
        self.set_list.pop(0)



        # l = self.lane_track_MPC(x_state, self.N_mpc)
        # x_pred, u_pred, solver_status = self.strategy_MPC(x_state, self.N_mpc)
        x_state[0] += 0.05
        try:
            # x_pred, u_pred, solver_status = self.lane_track_MPC_casadi(x_state, self.N_mpc)
            if x_state[0] < 1:
                l = self.lane_track_MPC_accel(x_state, self.N_mpc)
            else:
                l = self.lane_track_MPC_casadi(x_state, self.N_mpc)
            # return x_pred, u_pred, solver_status
            # print(l)
            return l
        except Exception as e:
            print(e)
        # if solver_status == False:
        #
        #     # replace the last set with the empty set
        #     self.set_list.pop(-1)
        #     # self.set_list.append([0, 0.25, 0.3, 0.25])
        #     self.set_list.append([])
        #
        #     self.N -= 1
        #     if self.N <= 17:
        #         # apply safety controller
        #         print('Infeasible strategy -- Using safety controller')
        #         x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, self.N_mpc)
        #
        #         if solver_status == False:
        #             print('infeasible safety controller - splitting horizon')
        #             x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, int(self.N_mpc / 2))
        #             if solver_status == False:
        #                 print('infeasible safety controller - splitting horizon')
        #                 x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, 1)
        #                 if solver_status == False:
        #                     print('giving up.')
        #         x_pred = np.hstack((x_pred, np.zeros((6, self.N + 1 - np.shape(x_pred)[1]))))
        #
        #     else:
        #         print('Infeasible strategy -- shortening horizon')
        #         # print(x_pred)
        #         # print(u_pred)
        #         x_pred, u_pred, solver_status = self.strategy_MPC(x_state, self.N)
        #
        #         if solver_status == False:
        #             x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, self.N_mpc)
        #
        #             if solver_status == False:
        #                 x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, int(self.N_mpc / 4))
        #
        #                 if solver_status == False:
        #                     x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, 1)
        #                     if solver_status == False:
        #                         input('shortened horizon MPC was infeasible?')
        #
        #         print(f'XPRED: {x_pred}')
        #         print(f'UPRED: {u_pred}')
        #         try:
        #             x_pred = np.hstack((x_pred, np.zeros((6, self.N + 1 - np.shape(x_pred)[1]))))
        #         except:
        #             pass
        # else:
        #     # print('New feasible strategy found!')
        #     self.N = self.N_mpc
        #
        # return x_pred, u_pred
        #
        # if std_s > self.s_conf_thresh or std_ey > self.ey_conf_thresh:
        #     # add the empty set to set_list, but we won't use it in this iteration
        #     self.set_list.append([])
        #
        #     # remove the first set from set_list
        #     self.set_list.pop(0)
        #
        #     # shorten the horizon (to re-use the previous solution)
        #     self.N -= 1
        #     if self.N <= 17: #17
        #         print('Low confidence -- Using safety controller')
        #         x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, self.N_mpc)
        #
        #         if solver_status == False:
        #             print('infeasible safety controller - splitting horizon')
        #             x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, int(self.N_mpc/2))
        #             if solver_status == False:
        #                 print('infeasible safety controller - splitting horizon')
        #                 x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, 1)
        #                 if solver_status == False:
        #                     print('giving up.')
        #
        #         x_pred = np.hstack((x_pred, np.zeros((6, self.N + 1 - np.shape(x_pred)[1]))))
        #
        #     else:
        #         # we're just going to solve the "previous" MPC problem, which under no model uncertainty amounts to re-using the same inputs.
        #         print('Low confidence -- shortening horizon')
        #         x_pred, u_pred, solver_status = self.strategy_MPC(x_state, self.N)
        #
        #         if solver_status == False:
        #             x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, self.N_mpc)
        #             if solver_status == False:
        #                 x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, int(self.N_mpc/4))
        #                 if solver_status == False:
        #                     x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, 1)
        #                     if solver_status == False:
        #                         input('shortened horizon MPC was infeasible?')
        #
        #         x_pred = np.hstack((x_pred, np.zeros((6, self.N + 1 - np.shape(x_pred)[1]))))
        #
        # else:
        #
        #     # add the strategy set to the end of the set_list
        #     self.set_list.append(centers)
        #
        #     # remove the first set from set_list
        #     self.set_list.pop(0)
        #
        #     # high confidence set, so we evaluate it.
        #     x_pred, u_pred, solver_status = self.strategy_MPC(x_state, self.N_mpc)
        #
        #     if solver_status == False:
        #
        #         # replace the last set with the empty set
        #         self.set_list.pop(-1)
        #         self.set_list.append([])
        #
        #         self.N -= 1
        #         if self.N <= 17:
        #             # apply safety controller
        #             print('Infeasible strategy -- Using safety controller')
        #             x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, self.N_mpc)
        #
        #             if solver_status == False:
        #                 print('infeasible safety controller - splitting horizon')
        #                 x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, int(self.N_mpc/2))
        #                 if solver_status == False:
        #                     print('infeasible safety controller - splitting horizon')
        #                     x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, 1)
        #                     if solver_status == False:
        #                         print('giving up.')
        #             x_pred = np.hstack((x_pred, np.zeros((6, self.N + 1 - np.shape(x_pred)[1]))))
        #
        #         else:
        #             print('Infeasible strategy -- shortening horizon')
        #             # print(x_pred)
        #             print(u_pred)
        #             x_pred, u_pred, solver_status = self.strategy_MPC(x_state, self.N)
        #
        #             if solver_status == False:
        #                 x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, self.N_mpc)
        #
        #                 if solver_status == False:
        #                     x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, int(self.N_mpc/4))
        #
        #                     if solver_status == False:
        #                         x_pred, u_pred, solver_status = self.lane_track_MPC(x_state, 1)
        #                         if solver_status == False:
        #                             input('shortened horizon MPC was infeasible?')
        #
        #             print(f'XPRED: {x_pred}')
        #             print(f'UPRED: {u_pred}')
        #             x_pred = np.hstack((x_pred, np.zeros((6, self.N + 1 - np.shape(x_pred)[1]))))
        #
        #     else:
        #         print('New feasible strategy found!')
        #         self.N = self.N_mpc
        #
        #
        #
        #     return x_pred, u_pred