# Initializing parameters
t = 0 # Start time
T = 3000 # End time
num_steps = 20 # Number of time steps
dt = (T-t)/num_steps # Time step size
# R= radius of droplet
R=1
# B-value = concentration at the boundary
B_value =1000000
# O = outer dimension
o=10
# O_value = initial concentration outside
O_value =1000000
# Diffusion coefficients inside and outside
D_i=0.01
D_o=0.01


from re import X
from shutil import unregister_unpack_format
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

# Defining inner and outer domains and their space functions
domain_i, cell_markers_i, facet_markers_i = gmshio.read_from_msh("circle.msh", MPI.COMM_WORLD, gdim=2)
domain_o, cell_markers_o, facet_markers_o = gmshio.read_from_msh("outside.msh", MPI.COMM_WORLD, gdim=2)

V_i = fem.FunctionSpace(domain_i, ("CG", 1))
V_o = fem.FunctionSpace(domain_o, ("CG", 1))

# Setting initial conditions INSIDE
u_n_i = fem.Function(V_i)
init_i = fem.Function(V_i)
x_i = V_i.tabulate_dof_coordinates()
for i in range(x_i.shape[0]):
    midpoint = x_i[i,:]
    if np.isclose(midpoint[0]**2+midpoint[1]**2,R*R):
        init_i.vector.setValueLocal(i, B_value)
    else:
        init_i.vector.setValueLocal(i, 0)
u_n_i.interpolate(init_i)

# Setting initial conditions OUTSIDE
u_n_o = fem.Function(V_o)
init_o = fem.Function(V_o)
x_o = V_o.tabulate_dof_coordinates()
for i in range(x_o.shape[0]):
    midpoint = x_o[i,:]
    if np.isclose(midpoint[0]**2+midpoint[1]**2,o*o):
        init_o.vector.setValueLocal(i, O_value)
    else:
        init_o.vector.setValueLocal(i, 0)
u_n_o.interpolate(init_o)

# BOUNDARY condtion DOMAIN_I setting Neumann at the center and R (+ saving)
boundaries_i =[(1, lambda x: np.isclose(np.sqrt(x[0]**2+x[1]**2),R)),
            (2, lambda x: np.isclose(np.sqrt(x[0]**2+x[1]**2),0))]
facet_indices_i, facet_markers_i = [], []
fdim_i = domain_i.topology.dim - 1
for (marker, locator) in boundaries_i:
    facets_i = dolfinx.mesh.locate_entities(domain_i, fdim_i, locator)
    facet_indices_i.append(facets_i)
    facet_markers_i.append(np.full_like(facets_i, marker))
facet_indices_i = np.hstack(facet_indices_i).astype(np.int32)
facet_markers_i = np.hstack(facet_markers_i).astype(np.int32)
sorted_facets_i = np.argsort(facet_indices_i)
facet_tag_i = dolfinx.mesh.meshtags(domain_i, fdim_i, facet_indices_i[sorted_facets_i], facet_markers_i[sorted_facets_i])
domain_i.topology.create_connectivity(domain_i.topology.dim-1, domain_i.topology.dim)
with XDMFFile(domain_i.comm, "facet_tags_inside.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain_i)
    xdmf.write_meshtags(facet_tag_i)
ds_i = ufl.Measure("ds", domain=domain_i, subdomain_data=facet_tag_i)

# BOUNDARY condtion DOMAIN_O settng Neumann at the R and outer boundary
boundaries_o =[(3, lambda x: np.isclose(np.sqrt(x[0]**2+x[1]**2),R)),
            (4, lambda x: np.isclose(np.sqrt(x[0]**2+x[1]**2),o))]
facet_indices_o, facet_markers_o = [], []
fdim_o = domain_o.topology.dim - 1
for (marker, locator) in boundaries_o:
    facets_o = dolfinx.mesh.locate_entities(domain_o, fdim_o, locator)
    facet_indices_o.append(facets_o)
    facet_markers_o.append(np.full_like(facets_o, marker))
facet_indices_o = np.hstack(facet_indices_o).astype(np.int32)
facet_markers_o = np.hstack(facet_markers_o).astype(np.int32)
sorted_facets_o = np.argsort(facet_indices_o)
facet_tag_o = dolfinx.mesh.meshtags(domain_o, fdim_o, facet_indices_o[sorted_facets_o], facet_markers_o[sorted_facets_o])
domain_o.topology.create_connectivity(domain_o.topology.dim-1, domain_o.topology.dim)
with XDMFFile(domain_o.comm, "facet_tags_outside.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain_o)
    xdmf.write_meshtags(facet_tag_o)
ds_o = ufl.Measure("ds", domain=domain_o, subdomain_data=facet_tag_o)


# DEFINIG VARIATIONAL FORMULATION inside and 
f_=0
fi= Constant(domain_i, ScalarType(f_))
Di= Constant(domain_i, ScalarType(D_i))
ui, vi = ufl.TrialFunction(V_i), ufl.TestFunction(V_i)
Fi = ui*vi*ufl.dx + Di*dt*ufl.dot(ufl.grad(ui), ufl.grad(vi))*ufl.dx - (u_n_i + dt*fi)*vi*ufl.dx

fo= Constant(domain_o, ScalarType(f_))
Do= Constant(domain_o, ScalarType(D_o))
uo, vo = ufl.TrialFunction(V_o), ufl.TestFunction(V_o)
Fo = uo*vo*ufl.dx + Do*dt*ufl.dot(ufl.grad(uo), ufl.grad(vo))*ufl.dx - (u_n_o + dt*fo)*vo*ufl.dx
# creating file to store solution
xdmf_i = XDMFFile(domain_i.comm, "2D_Complete_i.xdmf", "w")
xdmf_o = XDMFFile(domain_o.comm, "2D_Complete_o.xdmf", "w")

# Finishing defining BOUNDARIES and apppending to F
class BoundaryCondition_i():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_no = Function(V_i)
            u_no.interpolate(values)
            facets = facet_tag_i.find(marker)
            dofs = locate_dofs_topological(V_i, fdim_i, facets)
            self._bc = dirichletbc(u_no, dofs)
        elif type == "Neumann":
                self._bc = ufl.inner(values, vi) * ds_i(marker)
        elif type == "Robin":
            self._bc = values[0] * ufl.inner(ui-values[1], vi)* ds_i(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc
    @property
    def type(self):
        return self._type

boundary_conditions_i = [BoundaryCondition_i("Neumann", 1, 0),
                        BoundaryCondition_i("Neumann", 2, 0)]
bcs_i = []
for condition in boundary_conditions_i:
    if condition.type == "Dirichlet":
        bcs_i.append(condition.bc)
    else:
        Fi += condition.bc

class BoundaryCondition_o():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_no = Function(V_o)
            u_no.interpolate(values)
            facets = facet_tag_i.find(marker)
            dofs = locate_dofs_topological(V_o, fdim_o, facets)
            self._bc = dirichletbc(u_no, dofs)
        elif type == "Neumann":
                self._bc = ufl.inner(values, vo) * ds_o(marker)
        elif type == "Robin":
            self._bc = values[0] * ufl.inner(uo-values[1], vo)* ds_o(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc
    @property
    def type(self):
        return self._type

boundary_conditions_o = [BoundaryCondition_o("Neumann", 1, 0),
                        BoundaryCondition_o("Neumann", 2, 0)]
bcs_o = []
for condition in boundary_conditions_o:
    if condition.type == "Dirichlet":
        bcs_o.append(condition.bc)
    else:
        Fo += condition.bc

# Finalizing VARIATIONAL PROBLEM
# inside
ai = fem.form(ufl.lhs(Fi))
Li = fem.form(ufl.rhs(Fi))

Ai = fem.petsc.assemble_matrix(ai)
Ai.assemble()
bi = fem.petsc.create_vector(Li)
uhi = fem.Function(V_i)

solver_i = PETSc.KSP().create(domain_i.comm)
solver_i.setOperators(Ai)
solver_i.setType(PETSc.KSP.Type.PREONLY)
solver_i.getPC().setType(PETSc.PC.Type.LU)

# write value of initial condition
xdmf_i.write_mesh(domain_i)
xdmf_i.write_function(u_n_i, t)

# outside
ao = fem.form(ufl.lhs(Fo))
Lo = fem.form(ufl.rhs(Fo))

Ao = fem.petsc.assemble_matrix(ao)
Ao.assemble()
bo = fem.petsc.create_vector(Lo)
uho = fem.Function(V_o)

solver_o = PETSc.KSP().create(domain_o.comm)
solver_o.setOperators(Ao)
solver_o.setType(PETSc.KSP.Type.PREONLY)
solver_o.getPC().setType(PETSc.PC.Type.LU)

# write value of initial condition
xdmf_o.write_mesh(domain_o)
xdmf_o.write_function(u_n_o, t)

### Trying to extract values at interface from outer
def interface(x:np.array) -> np.array:
     return np.isclose(x[0]*x[0] + x[1]*x[1],R*R)
dofs_o = fem.locate_dofs_geometrical(V_o, interface)
print(u_n_o.x.array[dofs_o][0])
###

# diff=fem.derivative(Fo, u_n_o, uo)
# print(diff)
# for n in range(num_steps):
#     # Update Diriclet boundary condition 
#     # u_exact.t+=dt
#     # u_D.interpolate(u_exact)
    
#     # Update the right hand side reusing the initial vector
#     with bi.localForm() as loc_bi:
#         loc_bi.set(0)
#     fem.petsc.assemble_vector(bi, Li)

#     with bo.localForm() as loc_bo:
#         loc_bo.set(0)
#     fem.petsc.assemble_vector(bo, Lo)
    
#     # Apply Dirichlet boundary condition to the vector
#     # fem.petsc.apply_lifting(b, [a], [[bcs]])
#     # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#     # fem.petsc.set_bc(b, [bcs])

#     # Solve linear problem
#     solver_i.solve(bi, uhi.vector)
#     uhi.x.scatter_forward()

#     solver_o.solve(bo, uho.vector)
#     uho.x.scatter_forward()

#     # Update solution at previous time step (u_n)
#     u_n_i.x.array[:] = uhi.x.array
#     u_n_o.x.array[:] = uho.x.array
    
#     # write value of computed solution at each step
#     t += dt
#     xdmf_i.write_function(u_n_i, t)
#     xdmf_o.write_function(u_n_o, t)

# xdmf.close()