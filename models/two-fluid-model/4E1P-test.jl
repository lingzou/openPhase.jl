using NonlinearSolve, LinearSolve, LinearAlgebra
using BenchmarkTools
using Printf
using CSV

# using MacroTools

include("./4E1P.jl")
using .TF4E1P

###########################################################################
# Setup constants/data structure to describe a 4E1P problem

const N = 48
const dx = 12.0 / N
const alpha_in = 0.2
const p_out = 1.0e5
const vl_in = 10.0
const dt = 0.01
const Nt = 50

cells = Array{Cell}(undef, N)
int_edges = Array{IntEdge}(undef, N-1)

inlet = vBndryEdge(name = "inlet", vl_bc = 10.0, vg_bc = 0.0, alpha_bc = 0.2)
outlet = pBndryEdge(name = "outlet", p_bc = 1.0e5, alpha_bc = 0.2)
for i = 1 : N-1
  int_edges[i] = IntEdge(name = string("edge_", i))
end

for i = 1 : N
  cells[i] = Cell(name = string("cell_", i), dx = 0.1)

  west_edge = (i == 1) ? inlet : int_edges[i-1]
  east_edge = (i == N) ? outlet : int_edges[i]
  makeConnection!(west_edge, cells[i], east_edge)
end

# debug
# for i = 1 : N
#   w_cell, w_edge, e_edge, e_cell = cells[i].w_cell, cells[i].w_edge, cells[i].e_edge, cells[i].e_cell
#   printName(w_cell)
#   printName(w_edge)
#   printName(cells[i])
#   printName(e_edge)
#   printName(e_cell)
#   println("")
# end
# println("*******************")
# for i = 1 : N-1
#   w_cell, w_edge, e_edge, e_cell = int_edges[i].w_cell, int_edges[i].w_edge, int_edges[i].e_edge, int_edges[i].e_cell
#   printName(w_edge)
#   printName(w_cell)
#   printName(int_edges[i])
#   printName(e_cell)
#   printName(e_edge)
#   println("")
# end
# printName(inlet.w_edge)
# printName(inlet.w_cell)
# printName(inlet.e_edge)
# printName(inlet.e_cell)
# println("")
# printName(outlet.w_edge)
# printName(outlet.w_cell)
# printName(outlet.e_edge)
# printName(outlet.e_cell)

extendConnection!.(cells)
extendConnection!.(int_edges)
extendConnection!.([inlet, outlet])


# println("Re-check")
# for i = 1 : N
#   w_cell, w_edge, e_edge, e_cell = cells[i].w_cell, cells[i].w_edge, cells[i].e_edge, cells[i].e_cell
#   printName(w_cell)
#   printName(w_edge)
#   printName(cells[i])
#   printName(e_edge)
#   printName(e_cell)
#   println("")
# end
# println("*******************")
# for i = 1 : N-1
#   w_cell, w_edge, e_edge, e_cell = int_edges[i].w_cell, int_edges[i].w_edge, int_edges[i].e_edge, int_edges[i].e_cell
#   printName(w_edge)
#   printName(w_cell)
#   printName(int_edges[i])
#   printName(e_cell)
#   printName(e_edge)
#   println("")
# end
# printName(inlet.w_edge)
# printName(inlet.w_cell)
# printName(inlet.e_edge)
# printName(inlet.e_cell)
# println("")
# printName(outlet.w_edge)
# printName(outlet.w_cell)
# printName(outlet.e_edge)
# printName(outlet.e_cell)




###########################################################################
# Definte the nonlinear 4E1P problem in the residual form

function tf_4e1p(du, u, pars)
  cells, inlet, outlet, int_edges = pars
  updateSolution!(inlet, u[1], u[2])
  for i = 1 : N
    updateSolution!(cells[i], u[4*i-1], u[4*i])
  end
  for i = 1 : N-1
    updateSolution!(int_edges[i], u[4*i+1], u[4*i+2])
  end
  updateSolution!(outlet, u[4*N+1], u[4*N+2])

  computeFluxes!(inlet)
  computeFluxes!.(int_edges)
  computeFluxes!(outlet)

  du[1], du[2] = momentum_eqn(inlet, dt, dx)
  # momentum_eqn!(inlet, dt, dx, du[1], du[2])
  # momentum_eqn!(inlet, dt, dx, @view du[1:2])
  for i = 1 : N
    du[4*i-1], du[4*i] = mass_eqn(cells[i], dt, dx)
    # mass_eqn!(cells[i], dt, dx, du[4*i-1], du[4*i])
    # mass_eqn!(cells[i], dt, dx, @view du[4*i-1:4*i])
  end
  for i = 1 : N-1
    du[4*i+1], du[4*i+2] = momentum_eqn(int_edges[i], dt, dx)
    # momentum_eqn!(int_edges[i], dt, dx, du[4*i+1], du[4*i+2])
    # momentum_eqn!(int_edges[i], dt, dx, @view du[4*i+1 : 4*i+2])
  end
  du[4*N+1], du[4*N+2] = momentum_eqn(outlet, dt, dx)
  # momentum_eqn!(outlet, dt, dx, du[4*N+1], du[4*N+2])
  # momentum_eqn!(outlet, dt, dx, @view du[4*N+1 : 4*N+2])
end



###########################################################################
# Initial conditions

u_old = zeros(N*4+2) # vl, vg, alpha, p
# u_old[1      : N+1]    .= 10     # vl
# u_old[N+2    : 2*N+2]  .= 0.0    # vg
# u_old[2*N+3  : 3*N+2]  .= 0.2    # alpha
# u_old[3*N+3  : 4*N+2]  .= 1e5    # p
u_old[1], u_old[2] = vl_in, 0.0
for i = 1 : N
  u_old[4*i-1], u_old[4*i] = alpha_in, p_out
end
for i = 1 : N-1
  # updateSolution!(int_edges[i], u[4*i+1], u[4*i+2])
  u_old[4*i+1], u_old[4*i+2] = vl_in, 0.0
end
# updateSolution!(outlet, u[4*N+1], u[4*N+2])
u_old[4*N+1], u_old[4*N+2] = vl_in, 0.0

initialize!.(cells, alpha_in, p_out)
initialize!.(int_edges, vl_in, 0.0)
initialize!.([inlet, outlet], vl_in, 0.0)


pars = cells, inlet, outlet, int_edges
# define the non-linear problem
prob = NonlinearProblem(tf_4e1p, u_old, pars; abstol = 1e-6, reltol = 1e-8)



###########################################################################
# March time steps to solve the transient problem

for i = 1 : Nt
  println("Solving time step: ", i)
  @time solve(prob, NewtonRaphson()) #; show_trace = Val(true), trace_level = TraceAll(2)))
  # global pars = u_to_sol(u_old)
  # global prob = remake(prob, p=pars)

  # @time (sol = solve(prob, NewtonRaphson()))
  # global u_old = sol.u
  # global pars = u_to_sol(u_old)
  # global prob = remake(prob, u0=u_old, p=pars)
  saveOldSolutions!.(cells)
  saveOldSolutions!.(int_edges)
  # saveOldSolutions!(inlet)
  saveOldSolutions!(outlet)

  u_old[1], u_old[2] = vl_in, 0.0
  for i = 1 : N
    u_old[4*i-1], u_old[4*i] = cells[i].alpha, cells[i].p
  end
  for i = 1 : N-1
    # updateSolution!(int_edges[i], u[4*i+1], u[4*i+2])
    u_old[4*i+1], u_old[4*i+2] = int_edges[i].vl, int_edges[i].vg
  end
  u_old[4*N+1], u_old[4*N+2] = outlet.vl, outlet.vg
end



###########################################################################
# write output as csv files

file = open("cell-vars.csv", "w")
@printf(file, "%s, %s, %s\n", "x[m]", "alpha[-]", "pressure[Pa]")
for i = 1 : N
  @printf(file, "%g, %.8e, %.12e\n", (i-0.5) * dx, cells[i].alpha, cells[i].p)
end
close(file)

file = open("edge-vars.csv", "w")
@printf(file, "%s, %s, %s\n","x[m]", "vl[m/s]", "vg[m/s]")
@printf(file, "%g, %.8e, %.8e\n", 0.0, inlet.vl, inlet.vg)
for i = 1 : N-1
  @printf(file, "%g, %.8e, %.8e\n", i * dx, int_edges[i].vl, int_edges[i].vg)
end
@printf(file, "%g, %.8e, %.8e\n", dx * N, outlet.vl, outlet.vg)
close(file)



###########################################################################
# (regression) test the final results with 'gold' results

gold_cell = CSV.read("gold/cell-vars.csv", CSV.Tables.matrix; header=true)
gold_edge = CSV.read("gold/edge-vars.csv", CSV.Tables.matrix; header=true)

alpha_final = zeros(N)
p_final = zeros(N)
vl_final = zeros(N+1)
vg_final = zeros(N+1)

for i = 1 : N
  alpha_final[i], p_final[i] = cells[i].alpha, cells[i].p
end
vl_final[1], vg_final[1] = inlet.vl, inlet.vg
for i = 2 : N
  vl_final[i], vg_final[i] = int_edges[i-1].vl, int_edges[i-1].vg
end
vl_final[N+1], vg_final[N+1] = outlet.vl, outlet.vg

println("norm2 error for alpha = $(norm(alpha_final - gold_cell[:, 2]))")
println("norm2 error for p = $(norm(p_final - gold_cell[:, 3]))")
println("norm2 error for vl = $(norm(vl_final - gold_edge[:, 2]))")
println("norm2 error for vg = $(norm(vg_final - gold_edge[:, 3]))")