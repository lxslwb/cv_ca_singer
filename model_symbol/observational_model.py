import sympy
from model_symbol.statevec_model import x_sym, x1_sym

h_sym = sympy.Matrix(x_sym[:6])
h_fun_sym = sympy.lambdify([x_sym], h_sym, modules='numpy')

h_jac_sym = h_sym.jacobian(x_sym)
h_jac_fun_sym = sympy.lambdify([x_sym], h_jac_sym, modules='numpy')


def h_fun(**kwargs):
    """

    :param kwargs:
    :return:
    """
    if 'x' not in kwargs:
        raise ValueError("'x' must be provided as keyword arguments.")
    return h_fun_sym(kwargs['x'][:, 0])


def h_jac_fun(**kwargs):
    if 'x' not in kwargs:
        raise ValueError("'x' must be provided as keyword arguments.")
    return h_jac_fun_sym(kwargs['x'][:, 0])
