import numpy as np
import matplotlib.pyplot as plt


def main():
    X = np.array([1, 1, 1])
    A = np.array([[2, 7, 6],
                  [9, 5, 1],
                  [4, 3, 8]])
    E_MACHINE = 2 * pow(10, -16)

    f1_analytic_par = {'A': A, 'val': sin_val, 'grad': sin_grad, 'hessian': sin_hessian}
    f1_val, f1_analytic_grad, f1_analytic_hess = f1(X, f1_analytic_par, nargout=3)
    f1_numeric_par = {'epsilon': E_MACHINE, 'f_par': f1_analytic_par, 'gradient': f1_grad}
    f1_numeric_grad, f1_numeric_hess = numdiff(f1, X, f1_numeric_par, nargout=2)

    f2_analytic_par = {"phi_val": sin_val, "h": exp_val,
                       "phi_g": sin_grad, "h'": exp_grad,
                       "phi_h": sin_hessian, "h''": exp_hess}
    f2_val, f2_analytic_grad, f2_analytic_hess = f2(X, f2_analytic_par, nargout=3)
    f2_numeric_par = {'epsilon': E_MACHINE, 'f_par': f2_analytic_par, 'gradient': f2_grad}
    f2_numeric_grad, f2_numeric_hess = numdiff(f2, X, f2_numeric_par, nargout=2)

    # Compute difference between numerical and analytical gradient
    f1_grad_error = abs(f1_analytic_grad - f1_numeric_grad)
    f2_grad_error = abs(f2_analytic_grad - f2_numeric_grad)

    # question 5.1
    names = ['X1', 'X2', 'X3']

    # graph 1

    plt.figure(figsize=(15, 5))
    plt.suptitle('f1 error of gradient numeric function by element index')
    plt.subplot(131)
    plt.bar(names, f1_analytic_grad)
    plt.xlabel('element index')
    plt.ylabel('analytic gradient')
    plt.title('f1')
    plt.subplot(132)
    plt.bar(names, f1_numeric_grad)
    plt.xlabel('element index')
    plt.ylabel('numeric gradient')
    plt.title('f1')
    plt.subplot(133)
    plt.bar(names, f1_grad_error)
    plt.xlabel('element index')
    plt.ylabel('error')
    plt.title('f1')
    plt.show()

    # graph 2
    plt.figure(figsize=(15, 5))
    plt.suptitle('f2 error of gradient numeric function by element index')
    plt.subplot(131)
    plt.bar(names, f2_analytic_grad)
    plt.xlabel('element index')
    plt.ylabel('analytic gradient')
    plt.title('f2')
    plt.subplot(132)
    plt.bar(names, f2_numeric_grad)
    plt.xlabel('element index')
    plt.ylabel('numeric gradient')
    plt.title('f2')
    plt.subplot(133)
    plt.bar(names, f2_grad_error)
    plt.xlabel('element index')
    plt.ylabel('error')
    plt.title('f2')
    plt.show()

    # question 2
    # graph 1
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.imshow(f1_analytic_hess)
    plt.title('analytical')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(f1_numeric_hess)
    plt.title('numerical')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(abs(f1_analytic_hess - f1_numeric_hess))
    plt.title('difference')
    plt.colorbar()

    # plt.tight_layout()
    plt.suptitle('f1 hessians and their difference')
    plt.show()

    # graph 2
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.imshow(f2_analytic_hess)
    plt.title('analytical')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(f2_numeric_hess)
    plt.title('numerical')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(abs(f2_analytic_hess - f2_numeric_hess))
    plt.title('difference')
    plt.colorbar()

    plt.suptitle('f1 hessians and their difference')
    plt.show()

    print("\n\n\nf2 analytical hessian=\n", f2_analytic_hess)
    print("\n\n\nf2 numerical hessian=\n", f2_numeric_hess)

    # part 3
    EPSILON_VALS = 1000
    EPSILON_INCREMENT = 1 / EPSILON_VALS

    # graph 3 for f1
    f1_grad_errors_by_eps = []

    for i in range(1, EPSILON_VALS):
        # f1_analytic_par = {'A': A, 'val': sin_val, 'grad': sin_grad, 'hessian': sin_hessian}
        f1_val, f1_analytic_grad, f1_analytic_hess = f1(X, f1_analytic_par, nargout=3)
        f1_numeric_par = {'epsilon': EPSILON_INCREMENT * i, 'f_par': f1_analytic_par,
                          'gradient': f1_grad}
        f1_numeric_grad, f1_numeric_hess = numdiff(f1, X, f1_numeric_par, nargout=2)
        f1_grad_errors_by_eps.append(max(abs(f1_numeric_grad-f1_analytic_grad)))

    plt.figure()
    plt.plot([x * EPSILON_INCREMENT for x in range(1, 1000)], f1_grad_errors_by_eps)
    plt.show()

    print('Minimal error in absolute value of gradient is achieved at epsilon',
          (np.argmin(f1_grad_errors_by_eps)+1)*EPSILON_INCREMENT)



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
    hess = np.dot(np.dot(matT, phi_H), mat)

    return phi_val, grad, hess


def f1_grad(x, par):
    '''
    nonlinear multivariate function, f1:R^n->R, f1(x) = phi(Ax)

    :param x: function argument, vector. x is R^(n x 1)
    :param par: A dictionary
            'A' : matrix R^(m x n)
            'val' : pointer to function phi, nonlinear multivariate function R^m -> R
            'grad' : pointer to gradient function of phi
    :return: f1'(x)
    '''
    useless, ret = f1(x, par, nargout=2)
    return ret



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
        If nargout is 1, returns f2(x)
        If nargout is 2, returns (f2(x), f2'(x))
        If nargout is 3, returns (f2(x), f2'(x), f2''(x))
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


def f2_grad(x, par):
    '''
    h2(x) = f(phi(x))
    f is a nonlinear multivariate function f2: R^n -> R

    :param x: function argument, vector. x is R^(n x 1)
    :param par: A dictionary
            "phi_val" : pointer to function phi, nonlinear multivariate function R^m -> R
            "h" : pointer to nonlinear scalar function, h: R -> R
            "phi_g" : pointer to gradient function of phi
            "h'" : pointer to derivative function of h
    :return: f2'(x)
    '''
    useless, ret = f2(x, par, nargout=2)
    return ret


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

    return np.array(hessian)


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
        'gradient' : gradient function of f
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
        right_sample = analytic_grad(x+epsilon*standard_base[i], par['f_par'])
        left_sample = analytic_grad(x-epsilon*standard_base[i], par['f_par'])
        h_i = (right_sample-left_sample)/(2*epsilon)
        hess.append(h_i)
    hess = np.array(hess)

    return grad, hess




if __name__ == "__main__":
    main()