import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Определяем систему ОДУ ===
def differential_equation(x, y):
    """
    y[0] = y(x)
    y[1] = y'(x)
    y'' - 6 y' + 8 y = 2 x^2 + x
    """
    dydx = np.zeros(2)
    dydx[0] = y[1]
    dydx[1] = 6*y[1] - 8*y[0] + 2*x**2 + x
    return dydx

# === Аналитическое решение ===
def analytical_solution(x):
    return (-35/8 * np.exp(2*x) + 33/16 * np.exp(4*x) +
            0.25*x**2 + 0.5*x + 5/16)

# === Параметры ===
x0 = 0.0
x_end = 2.0
y0 = np.array([-2.0, 0.0])  # y(0) = -2, y'(0) = 0

# === Численное решение через scipy.solve_ivp ===
x_eval = np.linspace(x0, x_end, 1000)
sol = solve_ivp(differential_equation, [x0, x_end], y0, t_eval=x_eval, method='RK45')

# === Аналитическое решение для сравнения ===
y_analytical = analytical_solution(x_eval)

# === Построение графика ===
plt.figure(figsize=(10,6))

# Численное решение y(x)
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label="Численное y(x)")

# Аналитическое решение
plt.plot(x_eval, y_analytical, 'r--', linewidth=2, label="Аналитическое y(x)")

plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Сравнение численного и аналитического решения")
plt.legend()
plt.grid(True)

# Сохраняем в файл
plt.savefig("plot.png", dpi=150, bbox_inches='tight')
print("График сохранён в plot.png")
