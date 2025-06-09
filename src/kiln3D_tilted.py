from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
# from mshr import *
from ufl import outer
import json
from petsc4py import PETSc
import sys

#Turn off logging
set_log_level(LogLevel.WARNING)
logging.getLogger("FFC").setLevel(logging.ERROR)
logging.getLogger("UFL_LEGACY").setLevel(logging.ERROR)
logging.getLogger("UFL").setLevel(logging.ERROR)

def read_dictionary(fname):
    with open(fname) as json_file:
        pars = json.load(json_file)
    return pars

def write_dictionary(fname,mydict):
    with open(fname, 'w') as fp:
        json.dump(mydict, fp,sort_keys=True, indent=4)

def write_state(mesh,q,fname,it=0,t=0):
    data = {
            'step': it,
            'time': t
    }
    write_dictionary(fname + '.json',data)

    f=HDF5File(mesh.mpi_comm(),fname  + '.h5', 'w')
    f.write(mesh,"mesh")
    f.write(q,"q",t)
    f.close()

def read_mesh(fname):
    mesh = Mesh()
    f = HDF5File(MPI.comm_world, fname + '.h5', 'r')
    f.read(mesh, "mesh", False)
    f.close()
    return mesh


# Parameters
T = 1.0             # final time
num_steps = 100     # number of time steps
τ  = T / num_steps  # time step size
ω  = 2*np.pi        # angular velocity (full rotation in T)
ε  = 0.1/2          # interface thickness
h0 = 0.4            # height of the granular bed
m  = 0.5            # mobility of the Allen-Cahn equation
ε_pre = 1e-3        # pressure stabilization parameter

μ_s    = 1.0        # solid viscosity
μ_g    = 1.0e-2     # gas viscosity
β_pen  = 1e+3       # penality normal flow
β_slip = 0.05        # slip tangential flow

# Cylinder parameters
radius = 1.0
height = 5.0
v_feed = Constant((0.0,0.0,5.0))

# # Full lower-half cylinder from z = -height to z = 0
# full_cylinder = Cylinder(Point(0.0, 0.0, -height),  # bottom center
#                          Point(0.0, 0.0, 0.0),      # top center
#                          radius, radius)

# # Subtract box that removes x < 0 part
# cutting_box = Box(Point(2*radius, -2*radius, -2*height),
#                   Point(0.0, 2*radius, 1.0))  # everything with x < 0

# # Perform subtraction
# half_cylinder = full_cylinder - cutting_box

# # Generate mesh
# mesh_resolution = 96
# mesh = generate_mesh(half_cylinder, mesh_resolution)
mesh = read_mesh("../meshes/mesh3D")
mesh = refine(mesh)

# Define boundary
tol     = 1E-4
def boundary_wall_bottom(x, on_boundary):
    rtol = 0.02
    return on_boundary and near(x[0]*x[0] + x[1]*x[1], 1.0, rtol)

def boundary_inflow(x, on_boundary):
    return on_boundary and near(x[2], -height, tol)

def boundary_outflow(x, on_boundary):
    return on_boundary and near(x[2], 0, tol)

facets = MeshFunction("size_t", mesh, 2)
AutoSubDomain(boundary_wall_bottom).mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)

#onefunc = interpolate(Constant(1.0), FunctionSpace(mesh, 'CG', 1))
#print("onefunc",assemble(onefunc*ds(1)),"=",np.pi*5)
#F = File("kiln3D_tilted/facets.pvd")
#F << facets
#sys.exit()


print('Mesh generated with', mesh.num_vertices(), 'vertices and', mesh.num_cells(), 'cells.')

# create initial mesh (half circular domain)

v_rot  = Expression(("omega * x[1]","-omega * x[0]","0"), degree=2, omega = ω)

# Space for Allen-Cahn problem
V = FunctionSpace(mesh, 'CG', 1)

# Space for Stokes problem
FE_u = VectorElement('CG', mesh.ufl_cell(), 1) # 2)
FE_p = FiniteElement('CG', mesh.ufl_cell(), 1)
Vs = FunctionSpace(mesh, MixedElement([FE_u, FE_p]))

# Define and solve variational problem for Stokes
def solve_stokes(vp_n,φ):
    vp,wq = Function(Vs),TestFunction(Vs)

    v,p = split(vp) # split vp into unknown functions velocity v and pressure p
    w,q = split(wq) # split wq into test functions for velocity w and pressure q

    μ      = μ_s * abs(φ) + μ_g * abs(1-φ) # viscosity function
    f_grav = Constant((-20.0,0,5.0)) * φ  # gravity function

    nn = FacetNormal(mesh)
    P  = Identity(3)-outer(nn,nn)

    v_par = P*v
    w_par = P*w

    # classical Stokes with RHS f_grav
    F_stokes  = 2 * μ * inner(sym(grad(v)), sym(grad(w))) * dx - p * div(w) * dx + q * div(v) * dx
    F_stokes += inner(grad(p),grad(q)) * ε_pre * dx  # pressure stabilization term
    F_stokes -= inner(f_grav, w) * dx

    # slip boundary terms and penalty to enforce v*n = 0 on boundary
    F_stokes -= β_slip * abs(φ) * inner(v_rot, w_par) * ds(1)
    F_stokes += β_pen  * inner(v, nn) * inner(w,nn) * ds(1)
    F_stokes += β_slip * abs(φ) * inner(v_par,w_par) * ds(1)

    bc = DirichletBC(Vs.sub(0), v_feed, boundary_inflow) # velocity boundary condition

    vp.assign(vp_n)
    solve(F_stokes == 0, vp,bc,solver_parameters={'newton_solver': {'linear_solver': 'mumps'}}) # 'lu'
    return vp

# Define and solve variational problem for convective Allen-Cahn
def solve_ac(φ_n, v,v1, τ):
    φ,ψ = Function(V),TestFunction(V)
    F_ac  = (φ - φ_n)/τ*ψ*dx + m*ε*inner(grad(φ), grad(ψ))*dx + 2*m/ε*φ*(2*φ*φ-3*φ+1)*ψ*dx
    F_ac += 0.5*(dot(v, grad(φ))+dot(v1, grad(φ_n))) * ψ * dx
    φ.assign(φ_n)
    bc = DirichletBC(V,φ_0, boundary_inflow) # phase field boundary condition
    solve(F_ac == 0, φ,bc,solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})
    return φ

#initial condition with tanh profile
φ_0 = Expression('(1 - tanh((x[0] + (1-h0))/epsilon))/2', degree=2, epsilon=ε, h0 = h0)
φ_n = interpolate(φ_0, V)
vp_n = Function(Vs)
vp_n1 = Function(Vs)

# Time-stepping loop
t = 0
Vout = VectorFunctionSpace(mesh, 'CG', 1)

for n in tqdm(range(num_steps)):
    # Update time
    t += τ

    # Solve for the velocity v
    vp_n1.assign(vp_n)
    vp = solve_stokes(vp_n,φ_n)
    # v,p = vp.split()

    # Solve the the phase field φ
    # φ = solve_ac(φ_n, v, τ)
    φ = solve_ac(φ_n,vp.sub(0),vp_n1.sub(0), τ)


    write_state(mesh,φ,'kiln3D/phi'+str(n),it=n,t=t)
    write_state(mesh,vp,'kiln3D/v'+str(n),it=n,t=t)

    # Update previous solution
    φ_n.assign(φ)
    vp_n.assign(vp)

print('Done.')