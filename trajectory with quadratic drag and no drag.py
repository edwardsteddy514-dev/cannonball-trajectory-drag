import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81              # Gravity (m/s^2)
rho = 1.225           # Air density at sea level (kg/m^3)
Cd = 0.47             # Drag coefficient for a sphere
r = 0.05              # Radius of the cannonball (m)
m = 5.0               # Mass of the cannonball (kg)
A = np.pi * (r**2)    # Cross-sectional area of the cannonball (m^2)
theta = 45.0          # Launch angle (degrees)

v0 = 120.0            # Initial velocity (m/s)
vx0 = v0 * np.cos(np.radians(theta))  # Initial x-velocity
vy0 = v0 * np.sin(np.radians(theta))  # Initial y-velocity
x0, y0 = 0.0, 0.0     # Initial position (m)
b = 0.5 * rho * Cd * A / m       # effective drag parameter (1/m)

#------------------------------------------------------------------------
#the ideal case without drag for comparison
#------------------------------------------------------------------------
t_flight = 2 * vy0 / g
t = np.linspace(0, t_flight, 400)   # 400 points from t=0 to landing
x = vx0 * t
y = vy0 * t - 0.5 * g * t**2

#------------------------------------------------------------------------
#trajectory with quadratic drag
#------------------------------------------------------------------------
def derivatives(t, state):
    x, y, vx, vy = state
    v = np.hypot(vx, vy)   # Total speed
    ax = -b * v * vx       # Acceleration in x due to drag
    ay = -g - b * v * vy   # Acceleration in y due to gravity and drag
    return [vx, vy, ax, ay]

def hit_ground(t, state):
    return state[1]  # Stop integration when y (height) is zero
hit_ground.terminal = True
hit_ground.direction = -1

t_span = (0, 100)  # Time span for the simulation
initial_state = [x0, y0, vx0, vy0]  # Initial state vector
t_eval = np.linspace(0, 100, 1000)  # Time points to evaluate the solution

solution = solve_ivp(derivatives, t_span, initial_state, method='RK45', events=hit_ground, t_eval=t_eval)

# --- Plot the Trajectory ---
x_vals = solution.y[0]
y_vals = solution.y[1]

# --- Print Results ---
print(f"Distance traveled: {x_vals[-1]:.2f} meters")
print(f"Flight time: {solution.t[-1]:.2f} seconds")


plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label=f'Quadratic Drag (v0 = {v0} m/s)', color='firebrick', linewidth=2)
plt.plot(x, y, label=f'No Drag (v0 = {v0} m/s)', color='navy', linewidth=2, linestyle='--')
plt.title(f"Cannonball Trajectory with Quadratic Air Resistance, Launch Angle = {theta}°")
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Height (m)')
plt.axhline(0, color='black', linewidth=1)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
