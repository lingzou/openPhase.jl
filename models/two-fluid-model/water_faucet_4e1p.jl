using NonlinearSolve, LinearSolve
using BenchmarkTools

const N = 48
const dx = 12.0 / N
const alpha_in = 0.2
const p_out = 1.0e5
const vl_in = 10.0
const dt = 0.01
const Nt = 50

# fluid properties
rho_l_fn(p) = 1e3 + 1e-7 * (p - 1e5)
rho_g_fn(p) = 0.5 + 1e-5 * (p - 1e5)

# mapping unknown vector to physical variables
function u_to_sol(u)
  vl    = u[1     :   N+1]      # N+1 edges, liquid velocity
  vg    = u[N+2   :   2*N+2]    # N+1 edges, gas velocity
  alpha = u[2*N+3 :   3*N+2]    # N   cells, void fraction
  p     = u[3*N+3 :   4*N+2]    # N   cells, pressure
  rho_l = rho_l_fn.(p)
  rho_g = rho_g_fn.(p)

  (vl, vg, alpha, p, rho_l, rho_g)
end

# The nonlinear problem
function water_faucet(du, u, pars)
  # current solutions
  vl, vg, alpha, pp, rho_l, rho_g = u_to_sol(u)
  # old solutions, p_old not needed
  vl_o, vg_o, alpha_o, _, rho_l_o, rho_g_o = pars

  # computer helper variables, mass_flux_l and mass_flux_g, with upwind method
  mass_flux_l = zeros(eltype(du), N+1)
  mass_flux_g = zeros(eltype(du), N+1)

  mass_flux_l[1] = vl_in * rho_l[1] * (1.0 - alpha_in)
  mass_flux_g[1] = 0.0
  for i = 2 : N
    mass_flux_l[i] = (vl[i] > 0) ? vl[i] * rho_l[i-1] * (1 - alpha[i-1]) : vl[i] * rho_l[i] * (1 - alpha[i])
    mass_flux_g[i] = (vg[i] > 0) ? vg[i] * rho_g[i-1] * (    alpha[i-1]) : vg[i] * rho_g[i] * (    alpha[i])
  end
  mass_flux_l[N+1] = (vl[N+1] > 0) ? vl[N+1] * rho_l[N] * (1 - alpha[N]) : vl[N+1] * rho_l_fn(p_out) * (1 - alpha_in)
  mass_flux_g[N+1] = (vg[N+1] > 0) ? vg[N+1] * rho_g[N] * (    alpha[N]) : vg[N+1] * rho_g_fn(p_out) * (    alpha_in)

  # 'residual' section
  # gas-phase and liquid-phase mass balance equations
  for i = 1 : N
    du[2*N+2+i] = (alpha[i] * rho_g[i] - alpha_o[i] * rho_g_o[i]) / dt + (mass_flux_g[i+1] - mass_flux_g[i]) / dx
    du[3*N+2+i] = ((1-alpha[i]) * rho_l[i] - (1-alpha_o[i]) * rho_l_o[i]) / dt + (mass_flux_l[i+1] - mass_flux_l[i]) / dx
  end

  # the two momentum equations
  du[1] = vl[1] - vl_in   # Dirichlet BC for vl = 10 (@ x = 0)
  du[N+2] = vg[1] - 0.0   # Dirichlet BC for vg = 0  (@ x = 0)
  for i = 2 : N+1
    dvl_dx = (vl[i] > 0) ? (vl[i] - vl[i-1]) / dx : (i == N+1) ? 0 : (vl[i+1] - vl[i]) / dx
    dvg_dx = (vg[i] > 0) ? (vg[i] - vg[i-1]) / dx : (i == N+1) ? 0 : (vg[i+1] - vg[i]) / dx
    dp_dx = (i == N+1) ? (p_out - pp[i-1]) * 2 / dx : (pp[i] - pp[i-1]) / dx
    rho_l_bar = (i == N+1) ? rho_l[N] : 0.5 * (rho_l[i-1] + rho_l[i])
    rho_g_bar = (i == N+1) ? rho_g[N] : 0.5 * (rho_g[i-1] + rho_g[i])

    du[i]     = (vl[i] - vl_o[i]) / dt + vl[i] * dvl_dx + dp_dx / rho_l_bar - 9.8
    du[N+1+i] = (vg[i] - vg_o[i]) / dt + vg[i] * dvg_dx + dp_dx / rho_g_bar - 9.8
  end
end


############# initial conditions #############
u_old = zeros(N*4+2) # vl, vg, alpha, p
u_old[1      : N+1]    .= 10     # vl
u_old[N+2    : 2*N+2]  .= 0.0    # vg
u_old[2*N+3  : 3*N+2]  .= 0.2    # alpha
u_old[3*N+3  : 4*N+2]  .= 1e5    # p

pars = u_to_sol(u_old)

# define the non-linear problem
prob = NonlinearProblem(water_faucet, u_old, pars; abstol = 1e-6, reltol = 1e-8)


# iterate with time steps to solve the transient problem
for i = 1 : Nt
  println("Solving time step: ", i)
  @time (global u_old = solve(prob, NewtonRaphson())) #; show_trace = Val(true), trace_level = TraceAll(2)))
  global pars = u_to_sol(u_old)
  global prob = remake(prob, p=pars)

  # @time (sol = solve(prob, NewtonRaphson()))
  # global u_old = sol.u
  # global pars = u_to_sol(u_old)
  # global prob = remake(prob, u0=u_old, p=pars)
end

println(u_old[2*N+3 : 3*N+2])