import sympy as sy
import numpy as np
from sympy import *
from sympy.abc import x
from sympy.abc import A


def main():
    data = np.array([2, 1, 8])[:, np.newaxis] #column vector
    mat = np.array([[2, 7, 6],
                    [9, 5, 1],
                    [4, 3, 8]])

    res1 = f1(data,mat)
    print("======> f1 output:")
    print("Value =\n",res1[0])
    print("Gradient =\n", res1[1])
    print("Hessian =\n", res1[2])

    res1 = f2(data, mat)
    print("======> f2 output:")
    print("Value =\n",res1[0])
    print("Gradient =\n", res1[1])
    print("Hessian =\n", res1[2])

def f1(data,matrix):
    Ax = np.dot(matrix,data)
    print("Ax=\n",Ax)
    val, grad, hess = Task3_1_func(Ax)
    print("grad=\n",grad)
    return val, np.dot(np.transpose(matrix),grad), hess

def f2(data,matrix):
    Ax = np.dot(matrix, data)
    phi_val, phi_grad, phi_hess = Task3_1_func(Ax)
    exp_val, exp_derv, exp_secd = Task3_2_func(phi_val)
    grad = phi_grad*exp_derv
    hess = phi_hess*(exp_derv + exp_secd)
    return exp_val, grad, hess

def Task3_1_func(x):
    # This function returns value, gradient, hessian
    # of expression f([x1,x2,x3])=sin(x1*x2*x3)
    v = np.ones(x.shape) # v -> column vector
    vT = np.transpose(v)
    val = np.sin(np.dot(vT, x))
    print("np.cos(59*31*75)=",np.cos(59*31*75))
    grad = v*np.cos(np.dot(vT, x))
    hess = np.dot(-v*np.sin(np.dot(vT, x)),vT)
    return val, grad, hess

def Task3_2_func(x):
    # This function return value, fist derivative, second derivative
    # of expression exp(x)
    return np.exp(x), np.exp(x), np.exp(x)


if __name__ == "__main__":
    main()