import numpy as np
import matplotlib.pyplot as plt


a0 = 0.5
b0 = 17

n0 = 30

x0 = np.arange(-1, 1, 0.001)


# Weierstrass function
def w(a, b, n, x):
    y = x * 0.
    for i in range(n+1):
        y += a**i * np.sin(b**i * np.pi * x)
    return y


y0 = w(a0, b0, n0, x0)
plt.plot(x0, y0)
plt.show()
