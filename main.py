import sympy as sp
import scipy.optimize as opt

# Define symbols
x1, x2, x3 = sp.symbols('x1 x2 x3')
t = sp.symbols('t')

# Define function
#f = -5 * x1**2 - 4 * x2**2 - 2 * x3 ** 2 + 2 * x1 * x2 - 2 * x2 * x3 + 8 * x2
f = 6*x1**2 + 4*x2**2+3*x3**2+3*x1*x2-x2*x3+10*x2

# Define gradient
grad_f = sp.Matrix([f]).jacobian([x1, x2, x3])

# Define hessian
hess_f = sp.hessian(f, [x1, x2, x3])



def needed_and_sufficient(func, grad, hess):
    method = 'needed and sufficient condition'

    needed = sp.solve(grad_f, [x1, x2, x3])

    def check_hessian(hessian):
        eigenvalues = hessian.eigenvals()
        numeric_eigenvalues = [eigenvalue.evalf().as_real_imag()[0] for eigenvalue in eigenvalues.keys()]
        print("Eigenvalues:")
        print(numeric_eigenvalues)
        positive = True
        for eigenvalue in numeric_eigenvalues:
            if eigenvalue < 0:
                positive = False
        return positive

    if check_hessian(hess_f):
        print('x* is a local minimum')
    else:
        print('x* is a local maximum')
    print(f"x* = {needed}")



def grad_desc(func, grad, hess):
    method = 'gradient descent with '
    x = sp.Matrix([0, 0, 0])
    epsilon = 0.001
    iter = 0
    while iter < 100:
        p_k = grad.subs({x1: x[0], x2: x[1], x3: x[2]})
        #find t_k
        t_k = 0.1
        while func.subs({x1: x[0] - t_k * p_k[0], x2: x[1] - t_k * p_k[1], x3: x[2] - t_k * p_k[2]}) > func.subs({x1: x[0], x2: x[1], x3: x[2]}):
            t_k *= 0.5
        x = [x[0] - t_k * p_k[0], x[1] - t_k * p_k[1], x[2] - t_k * p_k[2]]
        iter += 1
        if all(abs(p_k[i]) < epsilon for i in range(3)):
            break
        print(f"iteration: {iter} : {x}")
    return x



def grad_desc_opt(func, grad):
    method = 'gradient descent with optimal step'
    x = sp.Matrix([0, 0, 0])
    epsilon = 0.001
    for i in range(100):
        p_k = grad.subs({x1: x[0], x2: x[1], x3: x[2]})
        f_t = func.subs({x1: x[0] - t * p_k[0], x2: x[1] - t * p_k[1], x3: x[2] - t * p_k[2]})
        f_prime = sp.diff(f_t, t)
        t_opt = sp.solve(f_prime, t)
        x = [x[0].evalf() - t_opt[0] * p_k[0], x[1].evalf() - t_opt[0] * p_k[1], x[2].evalf() - t_opt[0] * p_k[2]]
        print(f"iteration: {i} : {x}")
        if all(abs(p_k[i]) < epsilon for i in range(3)):
            break
    return x

def coordinate_descent_with_line_search(func, grad):
    method = 'coordinate descent with line search'
    x = sp.Matrix([0, 0, 0])

    epsilon = 0.001
    for i in range(100):
        new_x = sp.Matrix([0, 0, 0])
        p_k = grad.subs({x1: x[0], x2: x[1], x3: x[2]})
        for k in range(3):
            f_t = func.subs({x1: x[0] - t * p_k[0], x2: x[1] - t * p_k[1], x3: x[2] - t * p_k[2]})
            f_prime = sp.diff(f_t, t)
            t_opt = sp.solve(f_prime, t)
            new_x[k] = x[k].evalf() - t_opt[0] * p_k[k]


        change = max(abs(new_x[i] - x[i]) for i in range(3))
        if change < epsilon:
            break
        x = new_x
        print(f"iteration: {i} : {x.values()}")
    return x

def newton(func, grad, hess):
    method = 'newton method'
    x = sp.Matrix([0, 0, 0])
    epsilon = 0.001
    for i in range(100):
        p_k = grad.subs({x1: x[0], x2: x[1], x3: x[2]})
        h_k = hess.subs({x1: x[0], x2: x[1], x3: x[2]})
        h_inv = h_k.inv()
        x = [x[0].evalf() - h_inv[0, 0] * p_k[0] - h_inv[0, 1] * p_k[1] - h_inv[0, 2] * p_k[2],
             x[1].evalf() - h_inv[1, 0] * p_k[0] - h_inv[1, 1] * p_k[1] - h_inv[1, 2] * p_k[2],
             x[2].evalf() - h_inv[2, 0] * p_k[0] - h_inv[2, 1] * p_k[1] - h_inv[2, 2] * p_k[2]]
        print(f"iteration: {i} : {x}")
        if all(abs(p_k[i]) < epsilon for i in range(3)):
            break
    return x


def flatcher_reeves(func):
    grad = sp.Matrix([sp.diff(func, x1), sp.diff(func, x2), sp.diff(func, x3)])
    x = sp.Matrix([0, 0, 0])
    pk_old = sp.Matrix([0, 0, 0])
    beta = sp.Matrix([0, 0, 0])
    for i in range(50):
        pk = grad.subs({x1: x[0], x2: x[1], x3: x[2]})
        if i == 0:
            beta = sp.Matrix([0, 0, 0])
        else:
            beta = (pk_old[0]**2 + pk_old[1]**2 + pk_old[2]**2) / (pk[0]**2 + pk[1]**2 + pk[2] ** 2) * pk_old
        pk_old = pk
        pk = pk + beta
        f_t = func.subs({x1: x[0].evalf() - t * pk[0], x2: x[1].evalf() - t * pk[1], x3: x[2].evalf() - t * pk[2]})
        f_diff = sp.diff(f_t, t)
        tk = sp.solve(f_diff, t)[0]
        x = [x[i].evalf() - tk*pk[i] for i in range(3)]
        print(i, x)
        if all(abs(pk[i]) < 1e-3 for i in range(3)):
            result = [(x1, x[0]), (x2, x[1]), (x3, x[2])]
            print("Минимум найден в точке: ", result)
            print("Значение функции в минимуме:", f.subs(result))
            break


flatcher_reeves(f)