import sympy as sym
import numpy as np

l_r = 0.13
l_f = 0.13
x0, x1, x2, x3, x4, u0, u1 = sym.symbols('x0 x1 x2 x3 x4 u0 u1')

def LinearizeModel():
    f = sym.Matrix([x2*sym.cos(x3+x4), x2*sym.sin(x3+x4),
                u0, (x2/l_r)*sym.sin(x4), sym.atan2((l_r/(l_r+l_f)*sym.tan(u1)), 1)])
    Asym = f.jacobian([x0, x1, x2, x3, x4])
    print(Asym)
    # Asub = Asym.subs([(x1, x_state[0]), (x2, x_state[1]), (x3, x_state[2]), (x4, x_state[3]), (x4, x_state[4]), (u0, u_state[0]), (u1, u_state[1])])

    Bsym = f.jacobian([u0, u1])
    # Bsub = Bsym.subs([(x1, x_state[0]), (x2, x_state[1]), (x3, x_state[2]), (x4, x_state[3]), (x4, x_state[4]), (u0, u_state[0]), (u1, u_state[1])])

    return Asym, Bsym

def substitute(A, B, x_state, u_state):
    
    Asub = A.subs([(x1, x_state[0]), (x2, x_state[1]), (x3, x_state[2]), (x4, x_state[3]), (x4, x_state[4]), (u0, u_state[0]), (u1, u_state[1])])
    Bsub = B.subs([(x1, x_state[0]), (x2, x_state[1]), (x3, x_state[2]), (x4, x_state[3]), (x4, x_state[4]), (u0, u_state[0]), (u1, u_state[1])])
    
    Anp = np.array(Asub).astype(float)
    Bnp = np.array(Bsub).astype(float)

    return Anp, Bnp
# Please submit the results of the following unit test
# theta2_bar = 0.1
# alpha_bar = 0.5
# A, B = LinearizeModel()
# print('A = ', A)
# print('B = ', B)


