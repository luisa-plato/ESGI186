# Importing FEniCS and the required libraries
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import mshr

# Turn off logging
set_log_level(LogLevel.ERROR)  # Only show errors

# Parameters
T = 1.0             # final time
num_steps = 100     # number of time steps
τ  = T / num_steps  # time step size
ω  = 2*np.pi        # angular velocity (full rotation in T)
ε  = 0.02           # interface thickness
h0 = 0.4            # height of the granular bed
m  = 0.5            # mobility of the Allen-Cahn equation

μ_s    = 1.0        # solid viscosity
μ_g    = 1.0e-2     # gas viscosity
β_pen  = 1e+4       # penality normal flow
β_slip = 0.05        # slip tangential flow
n_refine = 3        # Number of times to refine the mesh

# create initial mesh for a rectangular domain of length L
L = 5
mesh = Mesh('../meshes/rectangle.xml')
for i in range(n_refine):
    mesh = refine(mesh)  # Refine the mesh for better resolution
dim  = mesh.geometry().dim()

# Define the feeding velocity and phase field for the boundary condition at the inflow (left) boundary
v_in = Constant((5.0, 0.0))
phi_in = Expression('(1 - tanh((x[1] + (1-h0))/epsilon))/2', degree=2, epsilon=ε, h0 = h0)


# Define boundary
tol     = 1E-12
def boundary_top_n_bottom(x, on_boundary):
    return on_boundary and (near(x[1], -1, tol) or near(x[1], 1, tol))

def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)

def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], L, tol)

# Mark different part of the boundary for the definition of the variational problem
facets = MeshFunction("size_t", mesh, 1)
AutoSubDomain(boundary_top_n_bottom).mark(facets, 1)
AutoSubDomain(boundary_right).mark(facets, 2)
ds = Measure("ds", subdomain_data=facets)


# Space for Allen-Cahn problem
V = FunctionSpace(mesh, 'CG', 1)

# Space for Stokes problem
FE_u = VectorElement('CG', mesh.ufl_cell(), 2)
FE_p = FiniteElement('CG', mesh.ufl_cell(), 1)
Vs = FunctionSpace(mesh, MixedElement([FE_u, FE_p]))

# Define and solve variational problem for Stokes
def solve_stokes(vp_n, φ):
    vp,wq = Function(Vs),TestFunction(Vs)

    v,p = split(vp) # split vp into unknown functions velocity v and pressure p
    w,q = split(wq) # split wq into test functions for velocity w and pressure q

    μ      = μ_s * abs(φ) + μ_g * abs(1-φ) # viscosity function
    f_grav = Constant((5.0, -20.0)) * φ  # gravity function

    # Define the boundary condition
    bc_left = DirichletBC(Vs.sub(0), v_in, boundary_left)                                 # DirichletBC for the velocity field at the inflow (left) boundary
    bc_top_n_bottom = DirichletBC(Vs.sub(0).sub(1), Constant(0.0), boundary_top_n_bottom)   # Non-permeability conditions for the velocity field at the walls (top and bottom boundary)
    bcs = [bc_left, bc_top_n_bottom]                                                        # Collect boundary conditions

    # Define the variational problem

    #Define the normal vector and tangential vector
    nn = FacetNormal(mesh)
    tt = as_vector((nn[1], -nn[0]))

    # classical mixed formulation for the Stokes problem with RHS f_grav
    F_stokes  = 2 * μ * inner(sym(grad(v)), sym(grad(w))) * dx - p * div(w) * dx + q * div(v) * dx
    F_stokes -= inner(f_grav, w) * dx

    # slip with friction boundary terms at the walls (top and bottom)
    F_stokes += β_slip * abs(φ) * inner(v, tt) * inner(w, tt) * ds(1)

    # set initial guess for the Stokes solver to the previous velocity
    vp.assign(vp_n)

    # Solve the Stokes problem
    solve(F_stokes == 0, vp, bcs)

    return vp

# Define and solve variational problem for convective Allen-Cahn
def solve_ac(φ_n, v,v1, τ):
    φ,ψ = Function(V),TestFunction(V)

    #Define boundary conditions
    bc_left = DirichletBC(V, phi_in, boundary_left)
    
    F_ac  = (φ - φ_n)/τ*ψ*dx + m*ε*inner(grad(φ), grad(ψ))*dx + 2*m/ε*φ*(2*φ*φ-3*φ+1)*ψ*dx
    F_ac += 0.5*(dot(v, grad(φ))+dot(v1, grad(φ_n))) * ψ * dx

    #Outflow condition for right boundary
    #nn = FacetNormal(mesh)
    #F_ac -= dot(grad(φ) - φ * v, nn) * ψ * ds(2)

    # Set initial guess for the Allen-Cahn solver to the previous phase field
    φ.assign(φ_n)

    # Solve the Allen-Cahn problem
    solve(F_ac == 0, φ, bc_left)
    return φ

#initial condition with tanh profile
φ_0 = Expression('(1 - tanh((x[1] + (1-h0))/epsilon))/2', degree=2, epsilon=ε, h0 = h0)
φ_n = interpolate(φ_0, V)


# Create a VTK file for visualization
vtkfile_φ = File('rect_0/phi.pvd')
vtkfile_v = File('rect_0/velocity.pvd')

# Time-stepping loop
t = 0
vp_n = Function(Vs)
vp_n1 = Function(Vs)
Vout = VectorFunctionSpace(mesh, 'CG', 1)

for n in tqdm(range(num_steps)):
    # Update time
    t += τ
    vp_n1.assign(vp_n)  # Store previous velocity
    # Solve for the velocity v
    vp = solve_stokes(vp_n, φ_n)
    # v = vp.sub(0)

    # Solve the the phase field φ
    φ = solve_ac(φ_n,vp.sub(0),vp_n1.sub(0), τ)

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