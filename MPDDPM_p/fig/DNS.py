import numpy as np
import matplotlib.pyplot as plt

sh_value = 1
sl_value = 0.001
off_value = 5
T_value = 2000


def beta_t_function_np(t, sh=sh_value, sl=sl_value, off=off_value, T=T_value):
    return (((t + off) ** 2 * sh) + ((T + off) ** 2 - (t + off) ** 2) * sl) / ((T + off) ** 2)


def first_derivative_function_np(t, sh=sh_value, sl=sl_value, off=off_value, T=T_value):
    return 2 * (off + t) * (sh - sl) / ((T + off) ** 2)


t_values_np = np.linspace(0, T_value, 400)
# Calculate beta_t values using the corrected function
beta_t_values_np = beta_t_function_np(t_values_np)
first_derivative_values_np = first_derivative_function_np(t_values_np)

fig, ax1 = plt.subplots(figsize=(12, 7))
plt.grid(True)
# Plot the corrected beta_t function
ax1.set_xlabel('TimeStep (t)')
ax1.set_ylabel('β_t', color='tab:red', fontsize=14)
line1 = ax1.plot(t_values_np, beta_t_values_np, label='β_t', color='tab:red', linewidth=3)
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for the derivative
ax2 = ax1.twinx()
ax2.set_ylabel('dβ_t/dt', color='tab:blue', fontsize=14)
line2 = ax2.plot(t_values_np, first_derivative_values_np, label='dβ_t/dt', color='tab:blue', linewidth=3)
ax2.tick_params(axis='y', labelcolor='tab:blue')
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=4, fontsize=14)

# Title and grid
# plt.title('Corrected β_t and its Derivative over Time')
fig.tight_layout()

# plt.grid(True)
# plt.show()
plt.savefig('DNS.svg')
