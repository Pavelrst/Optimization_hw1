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
    res1 = f1(data, sin_dict)
    print("======> f1 output:")
    print("Value =\n",res1[0])
    print("Gradient =\n", res1[1])
    print("Hessian =\n", res1[2])


'''
x - data
par - value of phi, gradient value of phi, hessian_funct of phi, A

out - value of f, gradient vvalue of f, hessian value of f 
'''
def f1(x, par):
    mat = par['A']
    phi_val = par['val'](np.dot(mat,x))
    phi_g = par['grad'](np.dot(mat,x))
    phi_H = par['hessian'](np.dot(mat,x))

    matT = np.transpose(mat)
    grad = np.dot(matT,phi_g)
    hess = np.dot(np.dot(matT,phi_H)[:,:,0],mat)

    return phi_val, grad, hess


    return val, grad, hess

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
    assert len(x) == 3
    return np.exp(x)

def exp_hess(x):
    assert len(x) == 3
    return np.hess(x)

def Task3_2_func(x):
    assert len(x) == 3
    # This function return value, fist derivative, second derivative
    # of expression exp(x)
    assert len(x) == 3
    return exp_val(x), exp_grad(x), exp_hess(x)




if __name__ == "__main__":
    main()