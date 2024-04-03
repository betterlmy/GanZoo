import numpy as np
import matplotlib.pyplot as plt

sh_value = 0.1
sl_value = 0.001
off_value = 5
T_value = 2000


def first_derivative_function_np(t, sh=sh_value, sl=sl_value, off=off_value, T=T_value):
    return 2 * (off + t) * (sh - sl) / ((T + off) ** 2)


t_values_np = np.linspace(0, T_value, 400)
first_derivative_values_np = first_derivative_function_np(t_values_np)

# Plotting the first derivative using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(t_values_np, first_derivative_values_np, label=r'$\frac{d\beta_t}{dt}$')
# plt.title(r'Relationship between $\frac{d\beta_t}{dt}$ and $t$')
plt.xlabel('TimeStep (t)')
plt.ylabel(r'$\frac{d\beta_t}{dt}$')
plt.grid(True)
plt.legend()
plt.show()
