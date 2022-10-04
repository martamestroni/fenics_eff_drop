# Initializing parameters
t = 0 # Start time
T = 300 # End time
num_steps = 20 # Number of time steps
dt = (T-t)/num_steps # Time step size
R=1
B_value =1000000
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

domain, cell_markers, facet_markers = gmshio.read_from_msh("circle.msh", MPI.COMM_WORLD, gdim=2)

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
u_n = fem.Function(V)
init = fem.Function(V)
x = V.tabulate_dof_coordinates()
for i in range(x.shape[0]):
    midpoint = x[i,:]
    if np.isclose(midpoint[0]**2+midpoint[1]**2,R*R):
        init.vector.setValueLocal(i, B_value)
    else:
        init.vector.setValueLocal(i, 0)
u_n.interpolate(init)

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
# in case of analytical solution heat
# u_n = fem.Function(V)
# u_n.interpolate(u_exact)
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