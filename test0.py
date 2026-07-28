import numpy as np
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------
# Physical parameters (edit these to model a different cannonball)
# ---------------------------------------------------------------
g       = 9.81        # m/s^2
m       = 5.0         # kg, projectile mass
diam    = 0.10         # m, projectile diameter (a chunky 10 cm iron ball)
Cd      = 0.47         # drag coefficient of a sphere
rho     = 1.225        # kg/m^3, air density at sea level
 
v0      = 120.0        # m/s, muzzle velocity
theta0  = 42.0          # degrees, launch angle
 
A = np.pi * (diam / 2) ** 2      # cross-sectional area
b = 0.5 * rho * Cd * A / m       # effective drag parameter (1/m)
 
print(f"Drag parameter b = {b:.6f} 1/m")
print(f"(For reference, ballistic coefficient-like scale: 1/b = {1/b:.1f} m)")
 
# ---------------------------------------------------------------
# Ideal (no drag) case: closed form
# ---------------------------------------------------------------
theta = np.radians(theta0)
vx0, vy0 = v0 * np.cos(theta), v0 * np.sin(theta)
 
t_flight_ideal = 2 * vy0 / g
t_ideal = np.linspace(0, t_flight_ideal, 400)
x_ideal = vx0 * t_ideal
y_ideal = vy0 * t_ideal - 0.5 * g * t_ideal**2
 
range_ideal = vx0 * t_flight_ideal
max_h_ideal = vy0**2 / (2 * g)
 
# ---------------------------------------------------------------
# Drag case: RK4 numerical integration
# ---------------------------------------------------------------
def derivs(state):
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    ax = -b * v * vx
    ay = -g - b * v * vy
    return np.array([vx, vy, ax, ay])
 
def simulate(v0, theta0, dt=0.001, t_max=200.0):
    theta = np.radians(theta0)
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
    traj = [state.copy()]
    t = 0.0
    while state[1] >= 0 and t < t_max:
        k1 = derivs(state)
        k2 = derivs(state + 0.5 * dt * k1)
        k3 = derivs(state + 0.5 * dt * k2)
        k4 = derivs(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        traj.append(state.copy())
    traj = np.array(traj)
    # linear interpolation to land exactly at y = 0
    x1, y1 = traj[-2, 0], traj[-2, 1]
    x2, y2 = traj[-1, 0], traj[-1, 1]
    frac = y1 / (y1 - y2)
    x_land = x1 + frac * (x2 - x1)
    t_land = (t - dt) + frac * dt
    return traj, x_land, t_land
 
traj, range_drag, t_flight_drag = simulate(v0, theta0)
x_drag, y_drag = traj[:, 0], traj[:, 1]
max_h_drag = y_drag.max()
 
# ---------------------------------------------------------------
# Find optimal launch angle under drag (sweep) vs ideal (always 45)
# ---------------------------------------------------------------
angles = np.arange(20, 71, 1)
ranges_drag = []
for a in angles:
    _, r, _ = simulate(v0, a)
    ranges_drag.append(r)
best_angle_drag = angles[np.argmax(ranges_drag)]
 
# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------
print("\n--- Comparison at theta0 = {:.0f} deg, v0 = {:.0f} m/s ---".format(theta0, v0))
print(f"{'Metric':<20}{'Ideal':>15}{'With drag':>15}")
print(f"{'Range (m)':<20}{range_ideal:>15.1f}{range_drag:>15.1f}")
print(f"{'Max height (m)':<20}{max_h_ideal:>15.1f}{max_h_drag:>15.1f}")
print(f"{'Flight time (s)':<20}{t_flight_ideal:>15.2f}{t_flight_drag:>15.2f}")
print(f"\nOptimal launch angle for max range under drag: {best_angle_drag} deg (vs 45 deg ideal)")
 
# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
"""fig, (ax1) = plt.subplots(1, figsize=(13, 5))
 
ax1.plot(x_ideal, y_ideal, label="No drag (ideal)", lw=2, color="#1f77b4")
ax1.plot(x_drag, y_drag, label="Quadratic drag", lw=2, color="#d62728")
ax1.set_xlabel("Horizontal distance (m)")
ax1.set_ylabel("Height (m)")
ax1.set_title(f"Trajectory comparison ($v_0$={v0} m/s, $\\theta_0$={theta0}\u00b0)")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(bottom=0)"""

plt.figure(figsize=(13, 5))
plt.plot(x_ideal, y_ideal, label="No drag (ideal)", lw=2, color='navy', linestyle='--')
plt.plot(x_drag, y_drag, label="Quadratic drag", lw=2, color="#d62728")
plt.title(f"Trajectory comparison ($v_0$={v0} m/s, $\\theta_0$={theta0}\u00b0)")
plt.xlabel("Horizontal distance (m)")
plt.ylabel('Vertical Height (m)')
plt.axhline(0, color='black', linewidth=1)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
 
#ax2.plot(angles, ranges_drag, lw=2, color="#d62728", label="Range with drag")
#ax2.axvline(45, color="#1f77b4", lw=2, linestyle="--", label="Ideal optimum (45\u00b0)")
#ax2.axvline(best_angle_drag, color="#d62728", lw=1, linestyle=":",
#           label=f"Drag optimum ({best_angle_drag}\u00b0)")
#ax2.set_xlabel("Launch angle (deg)")
#ax2.set_ylabel("Range (m)")
#ax2.set_title("launch angle")
#ax2.legend()
#ax2.grid(alpha=0.3)
 
"""plt.tight_layout()
plt.show()"""