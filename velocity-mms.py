import enum
import itertools

import numpy as np
import pandas
import ufl
from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc


def pprint(*msg, verbose=True):
    if verbose:
        print(" ".join(map(str, msg)), flush=True)


class Formulation(enum.Enum):
    taylor_hood = enum.auto()
    grad_curl_sipg = enum.auto()
    grad_curl_ripg = enum.auto()


class ViscosityModel(enum.Enum):
    iso = enum.auto()
    smooth = enum.auto()


def tensor_jump(u, n):
    return ufl.outer(u, n)("+") + ufl.outer(u, n)("-")


def G_mult(G, tau):
    m, d = tau.ufl_shape
    return ufl.as_matrix([[ufl.inner(G[i, k, :, :], tau) for k in range(d)]
                          for i in range(m)])


def run_experiment(formulation: Formulation,
                   p: int, ele_n: int, viscosity_model: ViscosityModel,
                   cell_type: dolfinx.mesh.CellType,
                   diagonal_type: dolfinx.mesh.DiagonalType,
                   penalty_constant: float):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array((-1.0, -1.0)), np.array((1.0, 1.0))],
        [ele_n, ele_n],
        cell_type=cell_type,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=diagonal_type)

    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    if viscosity_model is ViscosityModel.iso:
        mu = dolfinx.fem.Constant(mesh, 1.0)
    elif viscosity_model is ViscosityModel.smooth:
        mu = 1 + ufl.sin(ufl.pi*x[0])**2 * ufl.sin(ufl.pi*x[1])**2

    vel_soln = ufl.as_vector((
        ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ))

    if formulation in (Formulation.grad_curl_sipg, Formulation.grad_curl_ripg):
        # Solution space degree one higher
        Vs = dolfinx.fem.FunctionSpace(mesh, ("CG", p + 1))
        phi_soln = dolfinx.fem.Function(Vs)
        phi_soln.interpolate(
            lambda x: 1.0/np.pi * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]))

        # Stream function space
        PSI = dolfinx.fem.FunctionSpace(mesh, ('CG', p))
        psi = ufl.TestFunction(PSI)
        phi = ufl.TrialFunction(PSI)

        def eps(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return 2 * mu * ufl.sym(ufl.grad(u))

        n = ufl.FacetNormal(mesh)

        # Forcing function
        f = -ufl.div(sigma(vel_soln))

        if formulation is Formulation.grad_curl_sipg:
            # Formulate standard SIPG problem
            h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
            beta = dolfinx.fem.Constant(mesh, penalty_constant * p**2) / h

            G = mu * ufl.as_tensor([[
                [[2, 0],
                 [0, 0]],
                [[0, 1],
                 [1, 0]]],
                [[[0, 1],
                  [1, 0]],
                 [[0, 0],
                  [0, 2]]]])

            def Bh(u, v):
                domain = ufl.inner(sigma(u), eps(v)) * ufl.dx
                interior = (
                    - ufl.inner(tensor_jump(u, n), ufl.avg(sigma(v)))
                    - ufl.inner(tensor_jump(v, n), ufl.avg(sigma(u)))
                    + ufl.inner(beta("+") * G_mult(ufl.avg(G), tensor_jump(u, n)),
                                tensor_jump(v, n))
                ) * ufl.dS
                exterior = (
                    - ufl.inner(ufl.outer(u, n), sigma(v))
                    - ufl.inner(ufl.outer(v, n), sigma(u))
                    + ufl.inner(beta * G_mult(G, ufl.outer(u, n)), ufl.outer(v, n))
                ) * ufl.ds
                return domain + interior + exterior

            def lh(v):
                domain = ufl.inner(f, v) * ufl.dx
                exterior = (
                    - ufl.inner(ufl.outer(vel_soln, n), sigma(v))
                    + ufl.inner(beta * G_mult(G, ufl.outer(vel_soln, n)),
                                ufl.outer(v, n))
                ) * ufl.ds
                return domain + exterior

        elif formulation is formulation.grad_curl_ripg:
            # Formulate RIPG problem
            d = mesh.geometry.dim

            # Inverse inequality constants
            if mesh.ufl_cell() == ufl.triangle:
                m_k = 3
                c_vol = ufl.CellVolume(mesh)
                f_area = ufl.FacetArea(mesh)
                def C_inv(p):
                    return ufl.sqrt((p + 1) * (p + d) / d * f_area / c_vol)
            elif mesh.ufl_cell() == ufl.quadrilateral:
                # ffcx currently does not support cell/facet measures of
                # iso-parametric cells degree p > 1
                import dolfin_dg.dolfinx
                c_vol = dolfin_dg.dolfinx.util.cell_volume_dg0(mesh)
                f_area = dolfin_dg.dolfinx.util.facet_area_avg_dg0(mesh)
                def C_inv(p):
                    return ufl.sqrt((p + 1)**2 * f_area / c_vol)
                m_k = 4

            # Used in facet norm estimation for penalty parameter
            def cellwise_maximum(f):
                V = f.function_space
                mesh = V.mesh
                dm = V.dofmap
                n_cells_local = mesh.topology.index_map(
                    mesh.topology.dim).size_local
                dofs_per_cell = len(dm.dof_layout.entity_closure_dofs(
                    mesh.topology.dim, 0))
                dtype = np.dtype((np.int32, dofs_per_cell)) \
                    if dofs_per_cell > 1 else np.int32
                local_cell_dofs = np.fromiter(
                    (dm.cell_dofs(k) for k in range(n_cells_local)),
                    dtype=dtype)

                cell_vals = f.vector.array_r[local_cell_dofs]
                if len(cell_vals.shape) == 1:
                    return cell_vals

                cell_max = np.max(cell_vals, axis=1)
                return cell_max

            # Estimate penalty parameter norms
            DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
            Vmu = dolfinx.fem.FunctionSpace(mesh, ("DG", 1))
            mu_dgp = dolfinx.fem.Function(Vmu)

            l_inf_f = dolfinx.fem.Function(DG0)
            mu_dgp.interpolate(
                dolfinx.fem.Expression(
                    abs(2 * mu), Vmu.element.interpolation_points()))
            l_inf_f.vector.array[:] = cellwise_maximum(mu_dgp)

            l_inf_k = dolfinx.fem.Function(DG0)
            mu_dgp.interpolate(
                dolfinx.fem.Expression(
                    abs((2 * mu)**-0.5), Vmu.element.interpolation_points()))
            l_inf_k.vector.array[:] = cellwise_maximum(mu_dgp)

            C_p = C_inv(p-2)
            alpha = C_p * l_inf_f * l_inf_k

            # Non-affine cell mappings have a prefactor of the transformation
            # Jacobian measure.
            if cell_type is dolfinx.mesh.CellType.quadrilateral:
                J_F = (ufl.Jacobian(mesh)**2)**0.5
                J_K_inv = (ufl.JacobianInverse(mesh)**2)**0.5

                J_F_f = dolfinx.fem.Function(DG0)
                mu_dgp.interpolate(
                    dolfinx.fem.Expression(
                        J_F, Vmu.element.interpolation_points()))
                J_F_f.vector.array[:] = cellwise_maximum(mu_dgp)

                J_K_k = dolfinx.fem.Function(DG0)
                mu_dgp.interpolate(
                    dolfinx.fem.Expression(
                        J_K_inv, Vmu.element.interpolation_points()))
                J_K_k.vector.array[:] = cellwise_maximum(mu_dgp)

                alpha *= J_F_f * J_K_k

            # Construct penalty parameter from the constant 𝛿
            delta = dolfinx.fem.Constant(mesh, penalty_constant)
            zeta = 1 / (delta * ufl.sqrt(m_k) * alpha)

            # Weighted average operator
            def avg_w(u):
                w_p = zeta("+") / (zeta("+") + zeta("-"))
                w_m = zeta("-") / (zeta("+") + zeta("-"))
                return w_p * u("+") + w_m * u("-")

            def Bh(u, v):
                domain = ufl.inner(sigma(u), eps(v)) * ufl.dx
                beta_int = (zeta("+") + zeta("-")) ** -2
                interior = (
                    - ufl.inner(tensor_jump(u, n), avg_w(sigma(v)))
                    - ufl.inner(tensor_jump(v, n), avg_w(sigma(u)))
                    + ufl.inner(beta_int * tensor_jump(u, n),
                                tensor_jump(v, n))
                ) * ufl.dS
                beta_ext = zeta ** -2
                exterior = (
                    - ufl.inner(ufl.outer(u, n), sigma(v))
                    - ufl.inner(ufl.outer(v, n), sigma(u))
                    + ufl.inner(beta_ext * ufl.outer(u, n), ufl.outer(v, n))
                ) * ufl.ds
                return domain + interior + exterior

            def lh(v):
                domain = ufl.inner(f, v) * ufl.dx
                beta_ext = zeta ** -2
                exterior = (
                    - ufl.inner(ufl.outer(vel_soln, n), sigma(v))
                    + ufl.inner(beta_ext * ufl.outer(vel_soln, n), ufl.outer(v, n))
                ) * ufl.ds
                return domain + exterior

        # Homogeneous BCs imposed on the stream function
        facets = dolfinx.mesh.locate_entities_boundary(
            mesh, dim=mesh.topology.dim-1,
            marker=lambda x: np.ones_like(x[0], dtype=np.int8))
        dofs = dolfinx.fem.locate_dofs_topological(
            PSI, mesh.topology.dim-1, facets)
        zero_bc = dolfinx.fem.dirichletbc(0.0, dofs, PSI)

        # Solve
        total_dofs = mesh.comm.allreduce(
            PSI.dofmap.index_map.size_local * PSI.dofmap.index_map_bs, MPI.SUM)
        if mesh.comm.rank == 0:
            pprint(f"{formulation}: "
                   f"Solving problem, Nele={ele_n}, total DoFs = {total_dofs}")

        problem = dolfinx.fem.petsc.LinearProblem(
            Bh(ufl.curl(phi), ufl.curl(psi)), lh(ufl.curl(psi)),
            bcs=[zero_bc],
            petsc_options={"ksp_type": "preonly",
                           "pc_type": "lu"})
        phi = problem.solve()
        phi.x.scatter_forward()
    elif formulation is Formulation.taylor_hood:
        # Standard Taylor Hood mixed element
        Ve = ufl.VectorElement("CG", mesh.ufl_cell(), p)
        Qe = ufl.FiniteElement("CG", mesh.ufl_cell(), p-1)
        We = ufl.MixedElement([Ve, Qe])
        W = dolfinx.fem.FunctionSpace(mesh, We)
        u, p_ = ufl.TrialFunctions(W)
        v, q = ufl.TestFunctions(W)

        # Bilinear formulation
        a = (
                ufl.inner(2*mu*ufl.sym(ufl.grad(u)), ufl.grad(v)) * ufl.dx
                - p_ * ufl.div(v) * ufl.dx
                - q * ufl.div(u) * ufl.dx
        )
        f = -ufl.div(2*mu*ufl.sym(ufl.grad(vel_soln)))
        L = ufl.inner(f, v) * ufl.dx

        # Impose BCs on subspaces for free slip condition
        facets_top_bot = dolfinx.mesh.locate_entities_boundary(
            mesh, dim=mesh.topology.dim-1,
            marker=lambda x: np.isclose(np.abs(x[1]), 1.0))
        facets_left_right = dolfinx.mesh.locate_entities_boundary(
            mesh, dim=mesh.topology.dim-1,
            marker=lambda x: np.isclose(np.abs(x[0]), 1.0))

        V_x = W.sub(0).sub(0).collapse()
        zero = dolfinx.fem.Function(V_x[0])
        dofs_lr = dolfinx.fem.locate_dofs_topological(
            (W.sub(0).sub(0), V_x[0]), mesh.topology.dim - 1, facets_left_right)
        zero_x_bc = dolfinx.fem.dirichletbc(zero, dofs_lr, W.sub(0).sub(0))

        V_y = W.sub(0).sub(1).collapse()
        zero = dolfinx.fem.Function(V_y[0])
        dofs_tb = dolfinx.fem.locate_dofs_topological(
            (W.sub(0).sub(1), V_y[0]), mesh.topology.dim - 1, facets_top_bot)
        zero_y_bc = dolfinx.fem.dirichletbc(zero, dofs_tb, W.sub(0).sub(1))

        # Pin pressure DoF in bottom left corner for solvable system
        Q = W.sub(1).collapse()
        zero_p = dolfinx.fem.Function(Q[0])
        dofs_p = dolfinx.fem.locate_dofs_geometrical(
            (W.sub(1), Q[0]),
            lambda x: np.isclose(x[0], -1.0) & np.isclose(x[1], -1.0))
        zero_p_bc = dolfinx.fem.dirichletbc(zero_p, dofs_p, W.sub(1))

        # Solve system
        total_dofs = mesh.comm.allreduce(
            W.dofmap.index_map.size_local * W.dofmap.index_map_bs, MPI.SUM)
        if mesh.comm.rank == 0:
            pprint(f"{formulation}: "
                   f"Solving problem, Nele={ele_n}, total DoFs = {total_dofs}")

        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[zero_x_bc, zero_y_bc, zero_p_bc],
            petsc_options={"ksp_type": "preonly",
                           "pc_type": "lu"})
        w = problem.solve()
        w.x.scatter_forward()
        u, p_ = ufl.split(w)

    # Compute error measures of our FE approximation
    def compute_error(functional):
        err = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form(functional)), op=MPI.SUM) ** 0.5
        return err

    h_measure = dolfinx.cpp.mesh.h(
        mesh._cpp_object, 2,
        np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hmax = mesh.comm.allreduce(h_measure.max(), op=MPI.MAX)

    # Tabulate error measures into pandas dataframe
    if formulation in (Formulation.grad_curl_sipg, Formulation.grad_curl_ripg):
        vel = ufl.curl(phi)

        def ip_norm(v):
            domain = ufl.inner(sigma(v), eps(v)) * ufl.dx
            beta_int = (zeta("+") + zeta("-")) ** -2
            interior = beta_int * ufl.inner(tensor_jump(v, n), tensor_jump(v, n)) * ufl.dS
            beta_ext = zeta ** -2
            exterior = beta_ext * ufl.inner(ufl.outer(v, n), ufl.outer(v, n)) * ufl.ds
            return domain + interior + exterior

        datum = pandas.DataFrame({
            "hmin": hmin,
            "hmax": hmax,
            "DoF": total_dofs,
            "𝛿": penalty_constant,
            "‖𝜙−𝜙ₕ‖L₂": compute_error((phi - phi_soln) ** 2 * ufl.dx),
            "‖𝜙−𝜙ₕ‖H¹": compute_error(ufl.grad(phi - phi_soln) ** 2 * ufl.dx),
            "‖𝜙−𝜙ₕ‖HCurl": compute_error(ufl.curl(phi - phi_soln) ** 2 * ufl.dx),
            "‖u−uₕ‖L₂": compute_error((vel - vel_soln) ** 2 * ufl.dx),
            "‖u−uₕ‖H¹": compute_error(ufl.grad(vel - vel_soln) ** 2 * ufl.dx),
            "‖u−uₕ‖HDiv": compute_error(ufl.div(vel - vel_soln) ** 2 * ufl.dx),
            "‖∇⋅uₕ‖L₂": compute_error(ufl.div(vel) ** 2 * ufl.dx),
            "BC norm 𝜙": compute_error((phi - phi_soln)**2*ufl.ds),
            "BC norm u": compute_error((ufl.curl(phi - phi_soln))**2*ufl.ds),
            "IPnorm": compute_error(ip_norm(vel - vel_soln))
                if formulation is Formulation.grad_curl_ripg else np.NaN
        }, index=[0])
    elif formulation is Formulation.taylor_hood:
        datum = pandas.DataFrame({
            "hmin": hmin,
            "hmax": hmax,
            "DoF": total_dofs,
            "‖u−uₕ‖L₂": compute_error((u - vel_soln) ** 2 * ufl.dx),
            "‖u−uₕ‖H¹": compute_error(ufl.grad(u - vel_soln) ** 2 * ufl.dx),
            "‖u−uₕ‖HDiv": compute_error(ufl.div(u - vel_soln) ** 2 * ufl.dx),
            "‖∇⋅uₕ‖L₂": compute_error(ufl.div(u) ** 2 * ufl.dx)
        }, index=[0])

    return datum



if __name__ == "__main__":
    # A prefix for the output data files
    finame_prefix = "convergence"

    # Cases to cover in manufactured solution experiment
    ele_ns = [8, 16, 32]
    p_vals = [2]
    cell_types = [dolfinx.mesh.CellType.triangle]
    diagonal_types = [dolfinx.mesh.DiagonalType.left_right]
    viscosity_models = [ViscosityModel.smooth]
    formulations = [Formulation.grad_curl_ripg,
                    Formulation.grad_curl_sipg,
                    Formulation.taylor_hood]

    # Enumerate over possible cases
    for formulation, p, diagonal_type, cell_type, viscosity_model in itertools.product(
        formulations, p_vals, diagonal_types, cell_types, viscosity_models):

        # Set the value of δ as required by the numerical scheme
        penalty_val = 10.0 if formulation is Formulation.grad_curl_sipg else 2**0.5

        # Build pandas tabulation for each case
        df = pandas.DataFrame()
        for ele_n in ele_ns:
            datum = run_experiment(
                formulation, p, ele_n, viscosity_model,
                cell_type, diagonal_type, penalty_val)
            df = pandas.concat((df, datum), ignore_index=True)

        # Save recorded data to disk and output to std out
        if MPI.COMM_WORLD.rank == 0:
            import pathlib
            parent_dir = pathlib.Path("mms_data_testing")
            if not parent_dir.exists():
                parent_dir.mkdir()
            finame = (
                f"vel_mms_{finame_prefix}_p{p}_{formulation}_{viscosity_model}_{cell_type}"
                + (f"_{diagonal_type}" if cell_type is dolfinx.mesh.CellType.triangle else "")
                + f".csv"
            )
            with open(parent_dir / finame, "w") as fi:
                df.to_csv(fi)

            pandas.set_option('display.max_rows', df.shape[0])
            pandas.set_option('display.max_columns', df.shape[1])
            pandas.set_option('display.width', 180)

            print(f"{formulation}: Measured error norms")
            print(df)

            print(f"{formulation}: Convergence Rates")
            h_rate = df["hmin"]/df["hmin"].shift(-1)
            df_rate = df / df.shift(-1)
            error_rate = np.log(df_rate).divide(np.log(h_rate), axis="rows")
            print(error_rate.dropna(thresh=4))
