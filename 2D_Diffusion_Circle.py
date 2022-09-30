# Initializing parameters
t = 0 # Start time
T = 2 # End time
num_steps = 20 # Number of time steps
dt = (T-t)/num_steps # Time step size
alpha = 3
beta = 1.2

# Diffusion coefficient
D_=0.1


import numpy
import meshio
import gmsh
import sys
import pygmsh
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import FunctionSpace, Function, Constant, locate_dofs_geometrical, dirichletbc, form, Expression
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

# for squared domain
# nx, ny = 5, 5
# domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)


# 
# For circle domain
# Try1
# domain, cell_markers, facet_markers = gmshio.read_from_msh("circle1.msh", MPI.COMM_WORLD, gdim=3)
# Try2

# msh = meshio.read("circle1.msh")
# for cell in msh.cells:
#     quad_cells = cell.data
# for key in msh.cell_data_dict["gmsh:physical"].keys():s
#     if key == "quad":
#         quad_data = msh.cell_data_dict["gmsh:physical"][key]
# mesh =meshio.Mesh(points=msh.points,
#                            cells=[("quad", quad_cells)],
#                            cell_data={"name_to_read":[quad_data]})
# meshio.write("circle1.xdmf",mesh)

# with XDMFFile(MPI.COMM_WORLD, "circle1.xdmf", "r") as xdmf:
#        mesh = xdmf.read_mesh(name="Grid")

# Try3
# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
#     points = mesh.points[:,:2] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
#     return out_mesh

# msh = meshio.read("circle.msh")

# mesh = create_mesh(msh, "triangle", prune_z=True)
# meshio.write("circle.xdmf", mesh)


domain, cell_markers, facet_markers = gmshio.read_from_msh("circle.msh", MPI.COMM_WORLD, gdim=2)

# Try4
# rank = MPI.COMM_WORLD.rank

# gmsh.initialize()
# R=1 #Radius of domain
# gdim = 2  # Geometric dimension of the mesh
# model_rank = 0
# mesh_comm = MPI.COMM_WORLD
# if mesh_comm.rank == model_rank:
#     # Define geometry for background
#     background= gmsh.model.occ.addDisk(0, 0, 0, R, R)
#     gmsh.model.occ.synchronize()
#     background_surfaces = []
#     gmsh.model.addPhysicalGroup(2, background_surfaces, tag=0)

#     gmsh.model.mesh.generate(gdim)
#     gmsh.write("trymesh.msh")
# domain = gmshio.read_from_msh("sphere1.msh", MPI.COMM_WORLD, gdim=3)

# domain, ct, _ = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
# gmsh.finalize()

# with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_meshtags(ct)

# Try5
# with XDMFFile(MPI.COMM_WORLD, "circle1.xdmf", "r") as xdmf:
#        domain = xdmf.read_mesh(name="Grid")

# This way it is reading the msh file but giving error when creating xdmf
# domain = gmshio.read_from_msh("tut", MPI.COMM_WORLD, 0, gdim=2)
# xdmf = XDMFFile(domain.comm, "Try_Circle.xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.close()

# resume here

V = fem.FunctionSpace(domain, ("CG", 1))


# defining exact solution
class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t
    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t
u_exact = exact_solution(alpha, beta, t)

# BOUNDARY CONDITION
u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# DEFINIG VARIATIONAL FORMULATION
u_n = fem.Function(V)
u_n.interpolate(u_exact)

# in case of analytical solution heat
# f = fem.Constant(domain, beta - 2 - 2 * alpha)
# in case diffusion problem
f_=0
f= Constant(domain, ScalarType(f_))

D= Constant(domain, ScalarType(D_))

# creating file to store solution
xdmf = XDMFFile(domain.comm, "2D_Diffusion_Circle.xdmf", "w")

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = u*v*ufl.dx + D*dt*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx - (u_n + dt*f)*v*ufl.dx
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
b = fem.petsc.create_vector(L)
uh = fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# write value of initial condition
xdmf.write_mesh(domain)
xdmf.write_function(u_n, t)

for n in range(num_steps):
    # Update Diriclet boundary condition 
    u_exact.t+=dt
    u_D.interpolate(u_exact)
    
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, L)
    
    # Apply Dirichlet boundary condition to the vector
    fem.petsc.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    
    # write value of computed solution at each step
    t += dt
    xdmf.write_function(u_n, t)

xdmf.close()

# Compute L2 error and error at nodes
V_ex = fem.FunctionSpace(domain, ("CG", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array-u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")