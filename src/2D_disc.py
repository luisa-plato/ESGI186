"""
2D Rotating Disc Simulation
===========================
Solves coupled Stokes-Allen-Cahn equations for granular flow in a rotating circular disc.
- Geometry: 2D circular domain (cross-section of rotating cylinder)
- Key features: Slip boundary conditions, pure rotation with angular velocity ω, no axial flow
"""

# Import FEniCS and the required libraries
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Turn off logging
set_log_level(LogLevel.ERROR)  # Only show errors

# Set parameter values
T = 2.0*4           # final time
num_steps = 500    # number of time steps
τ  = T / num_steps  # time step size
ω  = 2*np.pi        # angular velocity (full rotation in T)
ε  = 0.02           # interface thickness
h0 = 0.4            # height of the granular bed
m  = 0.005          # mobility of the Allen-Cahn equation

μ_s    = 1.0        # solid viscosity
μ_g    = 1e-2       # gas viscosity
β_pen  = 1e+4       # penality normal flow

# run1 
β_slip = 0.2           # slip tangential flow
fname  = 'disc2D_0_2'  # Output folder name

# run2
β_slip = 0.05          # slip tangential flow
fname  = 'disc2D_0_05' # Output folder name

n_refine = 3  # Number of times to refine the mesh

mesh = Mesh('../meshes/disc.xml')
for i in range(n_refine):
    mesh = refine(mesh)  # Refine the mesh for better resolution
dim  = mesh.geometry().dim()

# Define the rotating velocity field for the boundary condition
v_rot  = Expression(("omega * x[1]","-omega * x[0]"), degree=2, omega = ω)

FE_u = VectorElement('CG', mesh.ufl_cell(), 2)
FE_p = FiniteElement('CG', mesh.ufl_cell(), 1)
Vs = FunctionSpace(mesh, MixedElement([FE_u, FE_p])) # Mixed space for Stokes problem
V  = FunctionSpace(mesh, 'CG', 1) # Space for Allen-Cahn problem

# Define and solve variational problem for Stokes
def solve_stokes(vp_n,φ):
    vp,wq = TrialFunction(Vs),TestFunction(Vs)

    v,p = split(vp) # split vp into unknown functions velocity v and pressure p
    w,q = split(wq) # split wq into test functions for velocity w and pressure q

    μ      = μ_s * abs(φ) + μ_g * abs(1-φ) # viscosity function
    f_grav = Constant((0.0, -20.0)) * φ  # gravity function

    # Define the normal vector and tangential vector
    n = FacetNormal(mesh)
    P  = Identity(dim)-outer(n,n)

    # classical mixed-element formulation for the Stokes problem with RHS f_grav
    F_stokes  = 2 * μ * inner(sym(grad(v)), sym(grad(w))) * dx - p * div(w) * dx + q * div(v) * dx
    R_stokes  = inner(f_grav, w) * dx

    # slip with friction boundary terms and penalty to enforce v*n = 0 on boundary
    R_stokes += β_slip * abs(φ) * inner(P*v_rot,P*w) * ds
    F_stokes += β_pen  * inner(v, n) * inner(w,n) * ds
    F_stokes += β_slip * abs(φ) * inner(P*v,P*w) * ds

    # set initial guess for the Stokes solver to the previous velocity
    vv = Function(Vs)
    vv.assign(vp_n)

    #Solve the Stokes problem
    solve(F_stokes == R_stokes, vv)

    return vv

# Define and solve variational problem for convective Allen-Cahn
def solve_ac(φ_n, v,v1, τ):
    φ,ψ = Function(V),TestFunction(V)
    F_ac  = (φ - φ_n)/τ*ψ*dx + m*ε*inner(grad(φ), grad(ψ))*dx + 2*m/ε*φ*(2*φ*φ-3*φ+1)*ψ*dx
    F_ac += 0.5*(dot(v, grad(φ))+dot(v1, grad(φ_n))) * ψ * dx
    φ.assign(φ_n)
    solve(F_ac == 0, φ)
    return φ

#initial condition with tanh profile
φ_0 = Expression('(1 - tanh((x[1] + (1-h0))/epsilon))/2', degree=2, epsilon=ε, h0 = h0)
φ_n = interpolate(φ_0, V)

# Create a VTK file for visualization
vtkfile_φ = File(f'{fname}/phi.pvd')
vtkfile_v = File(f'{fname}/velocity.pvd')

# Time-stepping loop
t = 0
Vout = VectorFunctionSpace(mesh, 'CG', 1)
vp_n = Function(Vs)
vp_n1 = Function(Vs)
print(f"run simulation{fname}")

for n in tqdm(range(num_steps)):
    # Update time
    t += τ

    # Solve for the velocity v
    vp_n1.assign(vp_n)  # Store previous velocity
    vp = solve_stokes(vp_n,φ_n)

    # Solve the the phase field φ
    φ = solve_ac(φ_n, vp.sub(0),vp_n1.sub(0), τ)
    
    # Save the state every 50 steps
    vout = project(φ_n*vp.sub(0), Vout)
    φ_n.rename('phi','phi')
    vout.rename('v','v')

    # Save solution to file
    vtkfile_φ << (φ_n, t)
    vtkfile_v << (vout, t)

    # Update previous solution
    φ_n.assign(φ)
    vp_n.assign(vp)

print('Done.')