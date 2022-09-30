# Initializing parameters
t = 0 # Start time
T = 300 # End time
num_steps = 20 # Number of time steps
dt = (T-t)/num_steps # Time step size
alpha = 3
beta = 1.2
R=1
B_value =10000000
# Diffusion coefficient
D_=0.001

from re import X
import numpy as np
import meshio
import gmsh
import sys
import pygmsh
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import FunctionSpace, Function, Constant, locate_dofs_geometrical,locate_dofs_topological, dirichletbc, form, Expression
import dolfinx.mesh
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

# Try3 - KEEP TO WRITE FILE RO .xdmf FILE: good for inspecting the mesh
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

# You can solve either with (1)analytcal solution or (2) initial condition
# (1)defining exact solution
# class exact_solution():
#     def __init__(self, alpha, beta, t):
#         self.alpha = alpha
#         self.beta = beta
#         self.t = t
#     def __call__(self, x):
#         return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t
# u_exact = exact_solution(alpha, beta, t)

# (2) setting initial conditions
# class initial_condition:
#     def eval_cell(self,value, x):
#         if (np.sqrt(x[0]**2+x[1]**2)) == 1:
#             value[0]=100
#         else:
#             value[0]=0
# # u_exact = initial_condition(x,R,B_value)
# u_D = fem.Function(V)
# u_D.interpolate(initial_condition())
# x = V.tabulate_dof_coordinates()
# for i in x:
#     if on_boundary(x):
#         # print(np.sqrt(i[0]**2+i[1]**2))
#         u_n.vector.setValueLocal(i, 1000)
#     else:
#         # print(np.sqrt(i[0]**2+i[1]**2))
#         u_n.vector.setValueLocal(i, 0.0)


# def on_boundary(x):
#     return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)
# bound = fem.locate_dofs_geometrical(V, on_boundary)

x= ufl.SpatialCoordinate(domain)

# p = python3 2D_Diffusion_Circle.py
# expr = fem.Expression(p, V.element.interpolation_points())
u_n = fem.Function(V)
u_n.interpolate(lambda x: (B_value*np.trunc(np.sqrt(x[0]**2+x[1]**2))))
# x = V.tabulate_dof_coordinates()
# for i in x:
#     if np.sqrt(x[0]**2 + x[1]**2)==1:
#         # print(np.sqrt(i[0]**2+i[1]**2))
#         u_n.vector.setValueLocal(i, 1000)


        
# BOUNDARY CONDITION for analytical solution
# u_D = fem.Function(V)
# u_D.interpolate(u_exact)
# tdim = domain.topology.dim
# fdim = tdim - 1
# domain.topology.create_connectivity(fdim, tdim)
# boundary_facets = mesh.exterior_facet_indices(domain.topology)
# bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# BOUNDARY condtion settng Neumann at the center and boarders
boundaries =[(1, lambda x: np.isclose(np.sqrt(x[0]**2+x[1]**2),R)),
            (2, lambda x: np.isclose(np.sqrt(x[0]**2+x[1]**2),0))]

facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
with XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


# DEFINIG VARIATIONAL FORMULATION
# u_n = fem.Function(V)
# u_n.interpolate(u_exact)

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

# Finishing defining boundaries and apppending to F


class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_no = Function(V)
            u_no.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(V, fdim, facets)
            self._bc = dirichletbc(u_no, dofs)
        elif type == "Neumann":
                self._bc = ufl.inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * ufl.inner(u-values[1], v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

# Define the Dirichlet condition
boundary_conditions = [BoundaryCondition("Neumann", 1, 0),
                        BoundaryCondition("Neumann", 2, 0)
]

bcs = []
for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        F += condition.bc


a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

A = fem.petsc.assemble_matrix(a)
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
    # u_exact.t+=dt
    # u_D.interpolate(u_exact)
    
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, L)
    
    # Apply Dirichlet boundary condition to the vector
    # fem.petsc.apply_lifting(b, [a], [[bcs]])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # fem.petsc.set_bc(b, [bcs])

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    
    # write value of computed solution at each step
    t += dt
    xdmf.write_function(u_n, t)

xdmf.close()

# # Compute L2 error and error at nodes
# V_ex = fem.FunctionSpace(domain, ("CG", 2))
# u_ex = fem.Function(V_ex)
# u_ex.interpolate(u_exact)
# error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
# if domain.comm.rank == 0:
#     print(f"L2-error: {error_L2:.2e}")

# # Compute values at mesh vertices
# error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array-u_D.x.array)), op=MPI.MAX)
# if domain.comm.rank == 0:
#     print(f"Error_max: {error_max:.2e}")