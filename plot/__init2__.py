import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# === Определяем систему ОДУ ===
def differential_equation(x, y):
    """
    y'' - 3y' + 2y = x^x * sin(x)
    Преобразуем в систему:
    y[0] = y(x)
    y[1] = y'(x)
    y'' = 3y' - 2y + x^x * sin(x)
    """
    dydx = np.zeros(2)
    dydx[0] = y[1]  # y' = y[1]
    dydx[1] = 3*y[1] - 2*y[0] + (x**x) * np.sin(x)  # y'' = 3y' - 2y + x^x * sin(x)
    return dydx

# === Аналитическое решение ===
def analytical_solution(x):
    """
    y(x) = 3·exp(x) - 3·exp(2·x) + (x/10​−3/25​)sinx+(3x/10​+17/50​)cosx
    """
    return 3 * np.exp(x) - 3 * np.exp(2 * x)  + ((x/10 - 3/25) * np.sin(x) + (3*x/10+17/50) * np.cos(x))

# === Параметры ===
x0 = 0.0
x_end = 2.0
y0 = np.array([0.0, -3.0])  # y(0) = 0, y'(0) = -3

# === Численное решение через scipy.solve_ivp ===
x_eval = np.linspace(x0, x_end, 10000000)
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

plt.savefig("plot2.png", dpi=150, bbox_inches='tight')
print("График сохранён в plot2.png")
