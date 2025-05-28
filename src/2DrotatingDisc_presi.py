# Import FEniCS and the required libraries
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mshr
from ufl import outer

# Turn off logging
set_log_level(LogLevel.ERROR)  # Only show errors

#Set parameter values
T = 0.25           # final time
num_steps = 200      # number of time steps
τ  = T / num_steps  # time step size
ω  = 2*np.pi        # angular velocity (full rotation in T)
ε  = 0.02           # interface thickness
h0 = 0.4            # height of the granular bed
m  = 0.01           # mobility of the Allen-Cahn equation

μ_s    = 1.0        # solid viscosity
μ_g    = 1.0e-2     # gas viscosity
β_pen  = 1e+4       # penality normal flow
β_slip = 0.2        # slip tangential flow


# Create initial mesh (half circular domain)
domain = mshr.Circle(Point(0,0),1) # - mshr.Rectangle(Point(-1,0),Point(1,1))
mesh   = mshr.generate_mesh(domain,48)

# Define the rotating velocity field for the boundary condition
v_rot  = Expression(("omega * x[1]","-omega * x[0]"), degree=2, omega = ω)

# Space for Allen-Cahn problem
V = FunctionSpace(mesh, 'CG', 1)

# Space for Stokes problem
FE_u = VectorElement('CG', mesh.ufl_cell(), 2)
FE_p = FiniteElement('CG', mesh.ufl_cell(), 1)
Vs = FunctionSpace(mesh, MixedElement([FE_u, FE_p]))

# Define and solve variational problem for Stokes
def solve_stokes(vp_n,φ):
    vp,wq = TrialFunction(Vs),TestFunction(Vs)

    v,p = split(vp) # split vp into unknown functions velocity v and pressure p
    w,q = split(wq) # split wq into test functions for velocity w and pressure q

    μ      = μ_s * abs(φ) + μ_g * abs(1-φ) # viscosity function
    f_grav = Constant((0.0, -20.0)) * φ  # gravity function

    # Define the normal vector and tangential projection
    nn = FacetNormal(mesh)
    P  = Identity(2)-outer(nn,nn)

    v_par = P*v
    w_par = P*w

    # classical mixed-element formulation for the Stokes problem with RHS f_grav
    F_stokes  = 2 * μ * inner(sym(grad(v)), sym(grad(w))) * dx - p * div(w) * dx + q * div(v) * dx
    R_stokes  = inner(f_grav, w) * dx

    # slip with friction and penalized penetration boundary terms
    R_stokes += β_slip * abs(φ) * inner(v_rot, w_par) * ds
    F_stokes += β_pen  * inner(v, nn) * inner(w,nn) * ds
    F_stokes += β_slip * abs(φ) * inner(v_par, w_par) * ds

    # set initial guess for the Stokes solver to the previous velocity
    vv = Function(Vs)
    vv.assign(vp_n)

    #Solve the Stokes problem
    solve(F_stokes == R_stokes, vv)

    return vv

# Define and solve variational problem for convective Allen-Cahn
def solve_ac(φ_n, v, τ):
    φ,ψ = Function(V),TestFunction(V)
    F_ac  = (φ - φ_n)/τ*ψ*dx + m*ε*inner(grad(φ), grad(ψ))*dx + 2*m/ε*φ*(2*φ*φ-3*φ+1)*ψ*dx
    F_ac += dot(v, grad(φ)) * ψ * dx
    φ.assign(φ_n)
    solve(F_ac == 0, φ)
    return φ

#initial condition with tanh profile
φ_0 = Expression('(1 - tanh((x[1] + (1-h0))/epsilon))/2', degree=2, epsilon=ε, h0 = h0)
φ_n = interpolate(φ_0, V)

# Create a VTK file for visualization
vtkfile_φ = File('2DrotatingDisc/phi.pvd')
vtkfile_v = File('2DrotatingDisc/velocity.pvd')

# Time-stepping loop
t = 0
Vout = VectorFunctionSpace(mesh, 'CG', 1)
vp_n = Function(Vs)

for n in tqdm(range(num_steps)):
    # Update time
    t += τ

    # Solve for the velocity v
    vp = solve_stokes(vp_n,φ_n)

    # Solve the the phase field φ
    φ = solve_ac(φ_n, vp.sub(0), τ)

    # Prepare output
    vout = project(φ_n*vp.sub(0), Vout)
    φ_n.rename('phi','phi')
    vout.rename('v','v')

    # Save solution to file
    vtkfile_φ << (φ_n, t)
    vtkfile_v << (vout, t)

    # Update previous solution
    φ_n.assign(φ)
    vp_n.assign(vp)

