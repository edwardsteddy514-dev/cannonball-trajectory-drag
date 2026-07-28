Modelling Projectile Motion with Quadratic Drag: Ideal vs. Realistic Trajectories
A numerical comparison of drag-free and drag-affected cannonball trajectories using RK4 integration.

Abstract
Classical projectile motion, the parabola every physics student derives, assumes away air resistance entirely. This project models a cannonball's trajectory under quadratic drag (the physically correct regime for a fast-moving, human-scale projectile) and compares it against the idealized no-drag case. Because drag introduces a nonlinear coupling between the horizontal and vertical equations of motion, no closed-form solution exists; the drag case is solved with a fourth-order Runge-Kutta (RK4) integrator. The results show that drag doesn't just shrink the trajectory uniformly, it reshapes it, breaking the symmetric parabola into a skewed curve and shifting the optimal launch angle away from the textbook 45°.

1. Introduction
A projectile launched with the only force being gravity follows a symmetric parabolic arc, a result so clean it's often the first non-trivial physics derivation students see. But real projectiles move through air, and at the speeds involved (on the order of 100 m/s for an actual cannonball), the drag force is not negligible and does not scale linearly with velocity. It scales with the square of velocity, a regime known as quadratic drag, dominant whenever the Reynolds number is high (fast-moving, human-scale objects in air, as opposed to, say, dust settling in still air).
This project builds both models side by side:
	The ideal case — ordinary projectile motion, solved analytically.
	The drag case — the same problem with quadratic air resistance added, solved numerically.
and compares them on range, maximum height, flight time, and optimal launch angle.

2. Theoretical Background
2.1 The Ideal Case
With gravity as the only force, Newton's second law gives:
a_x=0  and  a_y= -g 
Integrating twice (with initial speed v_0 and launch angle θ gives the closed-form position equations:
x(t)=v_0  cos⁡θ t , 
y(t)=v_0  sin⁡θ t-1/2 gt^2 
Setting y(t)=0 and solving for the nonzero root gives the total flight time:
t_flight=(2v_0  sin⁡θ)/g 
from which range and maximum height follow directly:
Range=v_0  cos⁡〖θ 〗× t_flight ,
Max Height=〖(v_0  sin⁡θ)〗^2/2g 
This is the entire ideal-case solution, a handful of algebraic formulas, no iteration required.

2.2 The Drag Case
Quadratic drag introduces a force opposing the velocity vector, with magnitude proportional to v^2:
(F_d ) ⃗=-1/2 ρC_d Avv ⃗ ,
v=√(v_x^2+v_y^2 ) 
Where;
ρ is air density, 
C_d  the drag coefficient (≈0.47 for a sphere), and 
A the cross-sectional area.
Newton's second law split’s into:
 a_x=-bvv_x ,
a_y=-g-bvv_y 
Where; 
b=(ρC_d A)/2m 
The key difference from the ideal case: v=√(v_x^2+v_y^2 )  couples the two equations together. Neither can be solved independently of the other, and no elementary closed-form solution exists for this coupled nonlinear system, it must be integrated numerically.

3. Numerical Method: Fourth Order Runge-Kutta Method (RK4)
Define the state vector: 
 y ⃗=( x,y,〖 v〗_x,〖 v〗_y)  
Then the system becomes: 
x ̇=v_x ,
y ̇=v_y ,
a_x=-bvv_x ,
a_y=-g-bvv_y 
Derivative function:
f(y ⃗ )= ( v_x,v_y,〖 -bvv〗_x,〖 -g-bvv〗_y)  
A fourth-order Runge-Kutta (RK4) integrator advances this state forward in fixed timesteps $dt$. Rather than using a single slope estimate per step (as the simpler, less accurate Euler's method does), RK4 samples the slope four times per step and combines them as a weighted average:
k_1=f(y ⃗_n ) , k_2=f(y ⃗_n+1/2 dtk_1 ) ,  k_3=f(y ⃗_n+1/2 dtk_2 ) , k_4=f(y ⃗_n+1/2 dtk_3 )
y ⃗_(n+1)=y ⃗_n+ⅆt/6 (k_1+2k_2+2k_3+k_4 ) 
	k_1 is the slope at the start of the interval.
	k_2 , k_3 are two successive refinements of the slope at the interval's midpoint.
	k_4  is the slope at the (estimated) end of the interval.
This gives local error of O(ⅆt^5 ) per step, far better than Euler's of O(ⅆt^2 ), at the cost of four function evaluations instead of one. Because the drag dynamics are nonlinear but smooth (no stiffness), a fixed step size of ⅆt=0.001s proved accurate and stable.
Landing detection: since the integrator only checks the sign of y after each fixed step, the exact moment of impact is refined via linear interpolation between the last two points straddling y=0, rather than requiring an impractically small step size to land exactly on the ground.

4. Implementation
The full model was implemented in Python (NumPy for the numeric, Matplotlib for visualization). The core of the integrator:

 1. def derivs(state):
 2.     x, y, vx, vy = state
 3.     v = np.hypot(vx, vy)
 4.     ax = -b * v * vx
 5.     ay = -g - b * v * vy
 6.     return np.array([vx, vy, ax, ay])
 7.  
 8. def simulate(v0, theta0, dt=0.001, t_max=200.0):
 9.     theta = np.radians(theta0)
10.     state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
11.     traj = [state.copy()]
12.     t = 0.0
13.     while state[1] >= 0 and t < t_max:
14.         k1 = derivs(state)
15.         k2 = derivs(state + 0.5 * dt * k1)
16.         k3 = derivs(state + 0.5 * dt * k2)
17.         k4 = derivs(state + dt * k3)
18.         state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
19.         t += dt
20.         traj.append(state.copy())
21.     traj = np.array(traj)
22.     # linear interpolation to land exactly at y = 0
23.     x1, y1 = traj[-2, 0], traj[-2, 1]
24.     x2, y2 = traj[-1, 0], traj[-1, 1]
25.     frac = y1 / (y1 - y2)
26.     x_land = x1 + frac * (x2 - x1)
27.     t_land = (t - dt) + frac * dt
28.     return traj, x_land, t_land

 

The drag parameter b was computed from physical constants for a representative cannonball: a 5 kg iron sphere, 10 cm in diameter, at sea-level air density (ρ=1.225" " 〖"kg/m" 〗^3, C_d=0.47):
b=(ρC_d A)/2m=  (1.225 × 0.47 × π〖(0.05)〗^2)/(2 × 5.0)≈0.000452 m^(-1)  

4.1 Refinement: Adaptive-Step Integration with scipy.integrate.solve_ivp
The hand-rolled RK4 above uses a fixed step size (dt=0.001" " s) and locates the landing point after the fact via linear interpolation. A more robust implementation replaces both of these with tools purpose-built for the job: scipy.integrate.solve_ivp, using the adaptive-step RK45 method (Dormand-Prince), and a terminal event function that lets the solver find the exact landing time via root-finding rather than approximating it:
 1. from scipy.integrate import solve_ivp

 2. def derivatives(t, state):
 3.     x, y, vx, vy = state
 4.     v = np.hypot(vx, vy)   # Total speed
 5.     ax = -b * v * vx       # Acceleration in x due to drag
 6.     ay = -g - b * v * vy   # Acceleration in y due to gravity and drag
 7.     return [vx, vy, ax, ay]
 8.  
 9. def hit_ground(t, state):
10.     return state[1]  # Stop integration when y (height) is zero
11. hit_ground.terminal = True
12. hit_ground.direction = -1
13.  
14. t_span = (0, 100)  # Time span for the simulation
15. initial_state = [x0, y0, vx0, vy0]  # Initial state vector
16. t_eval = np.linspace(0, 100, 1000)  # Time points to evaluate the solution
17.  
18. solution = solve_ivp(derivatives, t_span, initial_state, method='RK45', events=hit_ground, t_eval=t_eval)
 

Two improvements over the fixed-step version: adaptive step sizing automatically takes finer steps where the trajectory curves quickly (e.g. near apex) and coarser steps where it doesn't, rather than using one fixed dt everywhere; and event-based termination finds the precise moment y=0 is crossed through dense-output root-finding, rather than the linear-interpolation approximation the hand-rolled version relies on. At the drag-optimal launch angle found earlier (42°), this implementation gives a range of 996.19 m and a flight time of 14.61 sec, consistent with the fixed-step results, confirming both implementations converge to the same physical answer.

5. Results
Simulating both models at v_0=120" " m/s and θ_0=45°:
Metric	Ideal (no drag)	With drag
Range	1467.9 m	993.4 m
Max height	367.0 m	292.8 m
Flight time	17.30 s	15.42 s

 
At the optimum angle of θ_0=42° :
Metric	Ideal (no drag)	With drag
Range	1459.8 m	997.9 m
Max height	328.6 m	264.3 m
Flight time	16.37 s	14.65 s

 
Sweeping launch angle from 20° to 70° reveals that the drag case's optimal launch angle for maximum range is 42°, not the ideal case's 45°.
6. Discussion
Three findings stand out:
Range loss is substantial but not catastrophic. A 32% reduction in range for this ball's size and speed illustrates why real ballistics tables, for artillery, snipers, or any fast projectile, cannot use the ideal parabola formula and expect useful accuracy.
The trajectory shape is asymmetric, not just smaller. The drag trajectory rises nearly as steeply as the ideal one early on (drag hasn't yet had time to act at high initial speed) but falls more steeply than it rose. This happens because drag continuously bleeds horizontal velocity throughout the flight, while gravity's vertical pull continues unopposed by an equivalent decelerating force on the way down.
45° is no longer optimal. Under drag, a flatter launch angle (42° here) outperforms 45°, because it front-loads horizontal velocity before drag has as much time to sap it, a small but real correction that matters for anything from historical artillery gunnery to modern ballistic calculators.
A single parameter governs everything. The drag parameter b=(ρC_d A)/2m , conceptually similar to the ballistic coefficient used in exterior ballistics, controls how far a given trajectory departs from the ideal case. Heavy, dense, streamlined projectiles (small A, large m) stay close to ideal; light or draggy ones diverge sharply.

7. Limitations and Future Work
This model holds air density and drag coefficient constant throughout the flight, which is a reasonable approximation for the altitudes involved here but breaks down for very high trajectories (density varies with altitude) or a wide range of Reynolds numbers (where C_d itself varies). Natural extensions include:
	Altitude-dependent air density via the barometric formula, for high-apogee trajectories.
	Magnus effect (spin-induced lift/curve), relevant for rifled projectiles.
	Wind as a background velocity field added to the relative velocity term.

8. Conclusion
Quadratic drag transforms projectile motion from a two-line algebraic result into a genuinely nonlinear system requiring numerical integration. Building both models side by side — one closed-form, one via RK4 — makes concrete something often left abstract in introductory physics: that the "ideal" parabola is a simplification whose accuracy depends entirely on how draggy the object is and how fast it's moving, and that even a well-understood numerical method like RK4 earns its complexity precisely where the physics stops being linear.
