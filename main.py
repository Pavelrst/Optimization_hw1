import numpy as np
import matplotlib.pyplot as plt


def main():
    X = np.array([1, 1, 1])
    A = np.array([[2, 7, 6],
                  [9, 5, 1],
                  [4, 3, 8]])
    E_MACHINE = 2 * pow(10, -16)

    f1_analytic_par = {'A': A, 'val': sin_val, 'grad': sin_grad}
    f1_val, f1_analytic_grad = f1(X, f1_analytic_par, nargout=2)
    f1_numeric_par = {'epsilon': E_MACHINE, 'f_par': f1_analytic_par}
    f1_numeric_grad = numdiff(f1, X, f1_numeric_par, nargout=1)

    f2_analytic_par = {"phi_val": sin_val, "h": exp_val,
                       "phi_g": sin_grad, "h'": exp_grad}
    f2_val, f2_analytic_grad = f2(X, f2_analytic_par, nargout=2)
    f2_numeric_par = {'epsilon': E_MACHINE, 'f_par': f2_analytic_par}
    f2_numeric_grad = numdiff(f2, X, f2_numeric_par, nargout=1)

    # Compute difference between numerical and analytical gradient
    f1_grad_error = abs(f1_analytic_grad - f1_numeric_grad)
    print("analytic grad=", f1_analytic_grad,
          "\nnumeric grad=", f1_numeric_grad,
          "\nerror=", f1_grad_error,
          "\n\n\n")

    f2_grad_error = abs(f2_analytic_grad - f2_numeric_grad)
    print("analytic grad=", f2_analytic_grad,
          "\nnumeric grad=", f2_numeric_grad,
          "\nerror=", f2_grad_error,
          "\n\n\n")

    names = ['X1', 'X2', 'X3']
    plt.figure(1)
    plt.subplot(121)
    plt.bar(names, f1_grad_error)
    plt.xlabel('element index')
    plt.ylabel('error')
    plt.title('f1')

    plt.subplot(122)
    plt.bar(names, f2_grad_error, color='r')
    plt.xlabel('element index')
    plt.ylabel('error')
    plt.title('f2')

    plt.suptitle('error of numeric function by element index')
    plt.show()

    # '''
    # A testing function for the excercise's required functions
    # :return:
    # '''
    # data = np.array([2, 1, 8])[:, np.newaxis]  # column vector
    # mat = np.array([[2, 7, 6],
    #                 [9, 5, 1],
    #                 [4, 3, 8]])
    # I = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1]])
    #
    # val, grad, hess = Task3_1_func(data)
    #
    # print("======> Task3_1_func output:")
    # print("Value =\n", val)
    # print("Gradient =\n", grad)
    # print("Hessian =\n", hess)
    #
    # sin_dict = {'A': I,
    #             'val': sin_val,
    #             'grad': sin_grad,
    #             'hessian': sin_hessian}
    # res1 = f1(data, sin_dict, nargout=3)
    # print("======> f1 output:")
    # print("Value =\n",res1[0])
    # print("Gradient =\n", res1[1])
    # print("Hessian =\n", res1[2])
    #
    # exp_dict = {"phi_val": sin_val,
    #             "phi_g": sin_grad,
    #             "phi_h": sin_hessian,
    #             "h": exp_val,
    #             "h'": exp_grad,
    #             "h''": exp_hess}
    #
    # res2 = f2(data, exp_dict, nargout=3)
    # print("======> f2 output:")
    # print("Value =\n", res2[0])
    # print("Gradient =\n", res2[1])
    # print("Hessian =\n", res2[2])
    #
    # numdiff_par = dict({'e': mat, 'fun_par': sin_dict, 'gradient': sin_grad})
    # a, b = numdiff(f1, data, numdiff_par, nargout=2)
    # print("\n\n\na=\n", a, "\nb=\n", b)


def f1(x, par, nargout=3):
    '''
    nonlinear multivariate function, f1:R^n->R, f1(x) = phi(Ax)

    :param x: function argument, vector. x is R^(n x 1)
    :param par: A dictionary ('struct') of parameters including:
        for every nargout:
            'A' : matrix R^(m x n)
            'val' : pointer to function phi, nonlinear multivariate function R^m -> R
        for nargout > 1:
            'grad' : pointer to gradient function of phi
        for nargout > 2:
            'hessian' (optional) : pointer to hessian function of phi
    :param nargout: number of arguments given, mimics nargout of matlab, can be 1,2 or 3
    :return: depends on nargout.
        If nargout is 1, returns f1(x)
        If nargout is 2, returns (f1(x), f1'(x))
        If nargout is 3, returns (f1(x), f1'(x), f1''(x))
    '''

    assert isinstance(x, np.ndarray)
    assert isinstance(nargout, int) and nargout in range(1, 4)
    assert isinstance(par, dict)

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


def f2(x, par, nargout=3):
    '''
    h2(x) = f(phi(x))
    f is a nonlinear multivariate function f2: R^n -> R

    :param x: function argument, vector. x is R^(n x 1)
    :param par: A dictionary ('struct') of parameters including:
        for every nargout:
            "phi_val" : pointer to function phi, nonlinear multivariate function R^m -> R
            "h" : pointer to nonlinear scalar function, h: R -> R
        for nargout > 1:
            "phi_g" : pointer to gradient function of phi
            "h'" : pointer to derivative function of h
        for nargout > 2:
            "phi_h" : pointer to hessian function of phi
            "h''" : pointer to second derivative function of h
    :param nargout: number of arguments given, mimics nargout of matlab, can be 1,2 or 3
    :return: depends on nargout.
        If nargout is 1, returns f(x)
        If nargout is 2, returns (f(x), f'(x))
        If nargout is 3, returns (f(x), f'(x), f''(x))
    '''

    assert isinstance(x, np.ndarray)
    assert isinstance(nargout, int) and nargout in range(1, 4)
    assert isinstance(par, dict)

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
    '''
    :param x: A 3x1 vector of floats
    :return: sin(x1*x2*x3)
    '''
    assert isinstance(x, np.ndarray)
    assert len(x) == 3
    return np.sin(x[0] * x[1] * x[2])


def sin_grad(x):
    '''
    :param x: A 3x1 vector of floats
    :return: gradient of sin(x1*x2*x3)
    '''
    assert isinstance(x, np.ndarray)
    assert len(x) == 3
    prod = x[0] * x[1] * x[2]
    grad = [np.cos(prod) * x[1] * x[2],
            np.cos(prod) * x[0] * x[2],
            np.cos(prod) * x[0] * x[1]]
    return np.array(grad)


def sin_hessian(x):
    '''
    :param x: A 3x1 vector of floats
    :return: hessian of sin(x1*x2*x3)
    '''
    assert isinstance(x, np.ndarray)
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
    assert isinstance(x, np.ndarray)
    return sin_val(x), sin_grad(x), sin_hessian(x)


def exp_val(x):
    return np.exp(x)


def exp_grad(x):
    return np.exp(x)


def exp_hess(x):
    return np.exp(x)


def Task3_2_func(x):
    assert isinstance(x, np.ndarray)
    assert len(x) == 3
    return exp_val(x), exp_grad(x), exp_hess(x)


def numdiff(myfunc, x, par, nargout=2):
    '''
    computes gradient and hessian of myfunc numerically

    :param myfunc: pointer to either of f1 or f2
    :param x: Input vector R^(mx1)
    :param par: a dictionary including keys:
        'epsilon' : The incerement of x
        'f_par' : parameters dictionary given to function
    :param nargout: Like nargout of matlab, can be 1 or 2
    :return: [gnum, Hnum]
        gnum : Numerical estimation of function gradient
        Hnum : Numerical estimation of function Hessian
    '''
    assert callable(myfunc)
    assert isinstance(x, np.ndarray)
    assert isinstance(par, dict)
    assert 'epsilon' in par.keys()
    assert isinstance(nargout, int)
    assert nargout in range(1, 3)

    epsilon_tot = par['epsilon']
    assert isinstance(epsilon_tot, float)
    max_abs_val_of_x = max(x.min(), x.max(), key=abs)
    epsilon = pow(epsilon_tot, 1 / 3) * max_abs_val_of_x

    standard_base = np.array(((1, 0, 0),
                              (0, 1, 0),
                              (0, 0, 1)))

    grad = []
    for i in range(0, len(x)):
        right_g_i = myfunc(x+epsilon*standard_base[i], par['f_par'], nargout=1)
        left_g_i = myfunc(x-epsilon*standard_base[i], par['f_par'], nargout=1)
        g_i = (right_g_i - left_g_i)/(2*epsilon)
        grad.append(g_i)
    grad = np.array(grad)

    if nargout == 1:
        return grad

    hess = []
    analytic_grad = par['gradient']
    assert callable(analytic_grad)
    for i in range(0, len(x)):
        h_i = (analytic_grad(x+epsilon*e[i])-analytic_grad(x-epsilon*e[i]))/(2*epsilon)
        hess.append(h_i)
    hess = np.array(hess)

    return grad, hess




if __name__ == "__main__":
    main()