import sympy as sy
import numpy as np
from sympy import *
# from sympy.abc import x
# from sympy.abc import A


def main():
    data = np.array([2, 1, 8])[:, np.newaxis] #column vector
    mat = np.array([[2, 7, 6],
                    [9, 5, 1],
                    [4, 3, 8]])
    I = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    val, grad, hess = Task3_1_func(data)

    print("======> Task3_1_func output:")
    print("Value =\n", val)
    print("Gradient =\n", grad)
    print("Hessian =\n", hess)

    sin_dict = {'A': I,
                'val': sin_val,
                'grad': sin_grad,
                'hessian': sin_hessian}
    res1 = f1(data, sin_dict, nargout=3)
    print("======> f1 output:")
    print("Value =\n",res1[0])
    print("Gradient =\n", res1[1])
    print("Hessian =\n", res1[2])

    exp_dict= {"phi_val": sin_val,
               "phi_g": sin_grad,
               "phi_h": sin_hessian,
               "h": exp_val,
               "h'": exp_grad,
               "h''": exp_hess}

    res2 = f2(data, exp_dict, nargout=3)
    print("======> f2 output:")
    print("Value =\n", res2[0])
    print("Gradient =\n", res2[1])
    print("Hessian =\n", res2[2])

'''
x - data
par - value_funct of phi, gradient_funct of phi, hessian_funct of phi, A

out - value of f, gradient vvalue of f, hessian value of f 
'''
def f1(x, par, nargout=3):
    assert nargout in range(1, 4)
    mat = par['A']
    phi_val = par['val'](np.dot(mat, x))

    if nargout == 1:
        return phi_val

    phi_g = par['grad'](np.dot(mat, x))
    matT = np.transpose(mat)
    grad = np.dot(matT, phi_g)

    if nargout == 2:
        return phi_val, grad

    phi_H = par['hessian'](np.dot(mat, x))
    hess = np.dot(np.dot(matT,phi_H)[:,:,0], mat)

    return phi_val, grad, hess

'''
x - data
par - value_funct of phi, gradient_funct of phi, hessian_funct of phi

out - value of f, gradient vvalue of f, hessian value of f 
'''
def f2(x, par, nargout=3):
    assert nargout in range(1, 4)
    phi_val = par["phi_val"](x)
    f2_val = par["h"](phi_val)
    if nargout == 1:
        return f2_val

    grad_phi = par["phi_g"](x)
    h_der = par["h'"](phi_val)
    f2_grad = grad_phi * h_der
    if nargout == 2:
        return f2_val, f2_grad

    hess_phi = par["phi_h"](x)
    h_sec_der = par["h''"](phi_val)
    f2_hess = hess_phi * (h_der + h_sec_der)

    return f2_val, f2_grad, f2_hess


def sin_val(x):
    assert len(x) == 3
    return np.sin(x[0] * x[1] * x[2])

def sin_grad(x):
    assert len(x) == 3
    prod = x[0] * x[1] * x[2]
    grad = [np.cos(prod) * x[1] * x[2],
            np.cos(prod) * x[0] * x[2],
            np.cos(prod) * x[0] * x[1]]
    return np.array(grad)

def sin_hessian(x):
    assert len(x) == 3
    prod = x[0] * x[1] * x[2]
    dx1dx1 = -np.sin(prod) * x[1] * x[1] * x[2] * x[2]
    dx1dx2 = -np.sin(prod) * x[0] * x[1] * x[2] * x[2] + np.cos(prod) * x[2]
    dx1dx3 = -np.sin(prod) * x[0] * x[1] * x[1] * x[2] + np.cos(prod) * x[1]

    dx2dx1 = dx1dx2
    dx2dx2 = -np.sin(prod) * x[0] * x[0] * x[2] * x[2]
    dx2dx3 = -np.sin(prod) * x[0] * x[0] * x[1] * x[2] + np.cos(prod) * x[0]

    dx3dx1 = dx1dx3
    dx3dx2 = dx2dx3
    dx3dx3 = -np.sin(prod) * x[0] * x[0] * x[1] * x[1]

    hessian = [[dx1dx1, dx1dx2, dx1dx3],
               [dx2dx1, dx2dx2, dx2dx3],
               [dx3dx1, dx3dx2, dx3dx3]]
    return hessian

def Task3_1_func(x):
    return sin_val(x), sin_grad(x), sin_hessian(x)

def exp_val(x):
    return np.exp(x)

def exp_grad(x):
    return np.exp(x)

def exp_hess(x):
    return np.exp(x)

def Task3_2_func(x):
    # This function return value, fist derivative, second derivative
    # of expression exp(x)
    assert len(x) == 3
    return exp_val(x), exp_grad(x), exp_hess(x)




if __name__ == "__main__":
    main()