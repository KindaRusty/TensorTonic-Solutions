def gradient_descent_quadratic(a, b, c, x, lr, steps):
    for i in range(steps):
        grad = 2*a*x + b
        x = x - lr*grad
    return float(x)