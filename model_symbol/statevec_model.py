import sympy

# cv
x1_sym = sympy.Matrix([sympy.symbols('x'), sympy.symbols('y'), sympy.symbols('z'),
                       sympy.symbols('\dot{x}'), sympy.symbols('\dot{y}'), sympy.symbols('\dot{z}')])

# ca, singer
x_sym = sympy.Matrix([sympy.symbols('x'), sympy.symbols('y'), sympy.symbols('z'),
                      sympy.symbols('\dot{x}'), sympy.symbols('\dot{y}'), sympy.symbols('\dot{z}'),
                      sympy.symbols('\ddot{x}'), sympy.symbols('\ddot{y}'), sympy.symbols('\ddot{z}')])
