import sympy
from model_symbol.statevec_model import x1_sym, x_sym

# cv

t_sym = sympy.symbols('{\Delta}t')

a_cv_sym = sympy.Matrix([[1, 0, 0, t_sym, 0, 0], [0, 1, 0, 0, t_sym, 0], [0, 0, 1, 0, 0, t_sym],
                         [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
a_cv_fun = sympy.lambdify([t_sym], a_cv_sym, modules='numpy')
f_cv_sym = a_cv_sym @ x1_sym
f_cv_jac_sym = f_cv_sym.jacobian(x1_sym)

f_cv_fun_sym = sympy.lambdify([x1_sym, t_sym], f_cv_sym, modules='numpy')
f_cv_jac_fun_sym = sympy.lambdify([x1_sym, t_sym], f_cv_jac_sym, modules='numpy')

# ca

a_ca_sym = sympy.Matrix([[1, 0, 0, t_sym, 0, 0, 1 / 2 * t_sym ** 2, 0, 0],
                         [0, 1, 0, 0, t_sym, 0, 0, 1 / 2 * t_sym ** 2, 0],
                         [0, 0, 1, 0, 0, t_sym, 0, 0, 1 / 2 * t_sym ** 2],
                         [0, 0, 0, 1, 0, 0, t_sym, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, t_sym, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, t_sym],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]
                         ])
a_ca_fun = sympy.lambdify([t_sym], a_ca_sym, modules='numpy')

f_ca_sym = a_ca_sym @ x_sym
f_ca_jac_sym = f_ca_sym.jacobian(x_sym)
f_ca_fun_sym = sympy.lambdify([x_sym, t_sym], f_ca_sym, modules='numpy')
f_ca_jac_fun_sym = sympy.lambdify([x_sym, t_sym], f_ca_jac_sym, modules='numpy')

# singer

alpha_sym = sympy.symbols('a')

a_singer_sym = sympy.Matrix(
    [[1, 0, 0, t_sym, 0, 0, (alpha_sym * t_sym - 1 + sympy.exp(-alpha_sym * t_sym)) / alpha_sym ** 2, 0, 0],
     [0, 1, 0, 0, t_sym, 0, 0, (alpha_sym * t_sym - 1 + sympy.exp(-alpha_sym * t_sym)) / alpha_sym ** 2, 0],
     [0, 0, 1, 0, 0, t_sym, 0, 0, (alpha_sym * t_sym - 1 + sympy.exp(-alpha_sym * t_sym)) / alpha_sym ** 2],
     [0, 0, 0, 1, 0, 0, (1 - sympy.exp(-alpha_sym * t_sym)) / alpha_sym, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, (1 - sympy.exp(-alpha_sym * t_sym)) / alpha_sym, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, (1 - sympy.exp(-alpha_sym * t_sym)) / alpha_sym],
     [0, 0, 0, 0, 0, 0, sympy.exp(-alpha_sym * t_sym), 0, 0],
     [0, 0, 0, 0, 0, 0, 0, sympy.exp(-alpha_sym * t_sym), 0],
     [0, 0, 0, 0, 0, 0, 0, 0, sympy.exp(-alpha_sym * t_sym)]
     ])
a_singer_fun = sympy.lambdify([t_sym, alpha_sym], a_singer_sym, modules='numpy')

f_singer_sym = a_singer_sym @ x_sym
f_singer_jac_sym = f_singer_sym.jacobian(x_sym)
f_singer_fun_sym = sympy.lambdify([x_sym, t_sym, alpha_sym], f_singer_sym, modules='numpy')
f_singer_jac_fun_sym = sympy.lambdify([x_sym, t_sym, alpha_sym], f_singer_jac_sym, modules='numpy')


def f_cv_fun(**kwargs):
    if 'x' not in kwargs and 't' not in kwargs:
        raise ValueError("Both 'x' and 't' must be provided as keyword arguments.")
    return f_cv_fun_sym(kwargs['x'][:, 0], kwargs['t'])


def f_cv_jac_fun(**kwargs):
    if 'x' not in kwargs or 't' not in kwargs:
        raise ValueError("Both 'x' and 't' must be provided as keyword arguments.")
    return f_cv_jac_fun_sym(kwargs['x'][:, 0], kwargs['t'])


def f_ca_fun(**kwargs):
    if 'x' not in kwargs and 't' not in kwargs:
        raise ValueError("Both 'x' and 't' must be provided as keyword arguments.")
    return f_ca_fun_sym(kwargs['x'][:, 0], kwargs['t'])


def f_ca_jac_fun(**kwargs):
    if 'x' not in kwargs or 't' not in kwargs:
        raise ValueError("Both 'x' and 't' must be provided as keyword arguments.")
    return f_ca_jac_fun_sym(kwargs['x'][:, 0], kwargs['t'])


def f_singer_fun(**kwargs):
    if 'x' not in kwargs and 't' not in kwargs:
        raise ValueError("Both 'x' and 't' must be provided as keyword arguments.")
    return f_singer_fun_sym(kwargs['x'][:, 0], kwargs['t'], kwargs['alpha'])


def f_singer_jac_fun(**kwargs):
    if 'x' not in kwargs or 't' not in kwargs:
        raise ValueError("Both 'x' and 't' must be provided as keyword arguments.")
    return f_singer_jac_fun_sym(kwargs['x'][:, 0], kwargs['t'], kwargs['alpha'])
