include("./Mod_TF4e1p.jl")
using .TF4E1P:westBndryCell, eastBndryCell, Cell4E1P, Edge4E1P, connect_edge!, extendEdgeConnection!, computeFluxes!, mass_eqn!

using NonlinearSolve, LinearSolve, LinearAlgebra


# cell1 = Cell4E1P(0.2, 100.0, nothing, nothing)
# cell2 = Cell4E1P(0.2, 100.0, nothing, nothing)
# edge  = Edge4E1P(1.0, 2.0, nothing, nothing)

# connect!(cell1, edge, cell2)

# println(cell1.edge_west)
# println(cell1.edge_east)
# println(cell2.edge_west)
# println(cell2.edge_east)
# println(edge.cell_west)
# println(edge.cell_east)

const N = 10
const dx = 12.0 / N
const alpha_in = 0.2
const p_out = 1.0e5
const vl_in = 10.0
const dt = 0.01

# fluid properties
rho_l_fn(p) = 1e3 + 1e-7 * (p - 1e5)
rho_g_fn(p) = 0.5 + 1e-5 * (p - 1e5)

function updateSolutions!(u, cells::Array{Cell4E1P}, edges::Array{Edge4E1P})
  # vl    = u[1     :   N+1]      # N+1 edges, liquid velocity
  # vg    = u[N+2   :   2*N+2]    # N+1 edges, gas velocity
  # alpha = u[2*N+3 :   3*N+2]    # N   cells, void fraction
  # p     = u[3*N+3 :   4*N+2]    # N   cells, pressure
  for i = 1 : N+1
    edges[i].vl    = u[i]
    edges[i].vg    = u[N+1 + i]
  end
  for i = 1 : N
    cells[i].alpha  = u[2*N+2 + i]
    cells[i].p      = u[3*N+2 + i]
    cells[i].rho_l  = rho_l_fn(cells[i].p)
    cells[i].rho_g  = rho_g_fn(cells[i].p)
  end

  # edges[1].mass_flux_l = edges[1].vl * cells[1].rho_l * (1.0 - alpha_in)
end


# wBC = westBndryCell("wBC", alpha_in, p_out)
wBC = westBndryCell(name = "wBC", alpha = alpha_in, p = p_out)
cells = Array{Cell4E1P}(undef, N)
edges = Array{Edge4E1P}(undef, N+1)
# eBC = eastBndryCell("eBC", alpha_in, p_out)
eBC = eastBndryCell(name = "eBC", alpha = alpha_in, p = p_out)

for i = 1 : N
  #cells[i] = Cell4E1P(string("cell_", i), dx, 0.2, 1e5, nothing, nothing, nothing, nothing)
  # cells[i] = Cell4E1P(string("cell_", i), dx, alpha_in, p_out)
  cells[i] = Cell4E1P(name = string("cell_", i), dx = dx, alpha = alpha_in)
end

for i = 1 : N+1
  west_cell = (i > 1) ? cells[i-1] : wBC
  east_cell = (i < N+1) ? cells[i] : eBC

  # west_cell = wBC # nothing
  # east_cell = eBC # nothing
  # if i > 1
  #   west_cell = cells[i-1]
  # end
  # if i < N+1
  #   east_cell = cells[i]
  # end

  # edges[i] = Edge4E1P(string("edge_", i), 10.0, 0.0, west_cell, east_cell, nothing, nothing)
  # edges[i] = Edge4E1P(string("edge_", i), vl_in, 0.0)
  edges[i] = Edge4E1P(name = string("edge_", i))
  connect_edge!(west_cell, edges[i], east_cell)
end

extendEdgeConnection!.(edges)


for i = 1 : N+1
  println(edges[i].vl)
  println(edges[i].vg)
  println(edges[i].mass_flux_l)
  println(edges[i].mass_flux_g)
end
for i = 1 : N
  println(cells[i].alpha)
  println(cells[i].p)
  println(cells[i].rho_l)
  println(cells[i].rho_g)
end
println("*************************")


function water_faucet(du, u, p)
  global cells
  global edges

  updateSolutions!(u, cells, edges)
  computeFluxes!.(edges)

  du[1] = edges[1].vl - 10.0
  du[1+N+1] = edges[1].vg - 0.0
  for i = 2 : N+1
    # du[i] = edges[i].vl - 10.0
    # du[i+N+1] = edges[i].vg - 0.0
    momentum_eqn!(du[i], du[i+N+1], edges[i], dt, dx)
  end
  for i = 1 : N
    # du[2*N+2 + i] = cells[i].alpha - 0.2
    # du[3*N+2 + i] = cells[i].p - 1.0e5
    mass_eqn!(du[2*N+2 + i], du[3*N+2 + i], cells[i], dt, dx)
  end

end

u_old = zeros(N*4+2) #zeros(N+N+1) #
prob = NonlinearProblem(water_faucet, u_old, nothing; abstol = 1e-6, reltol = 1e-8)

sol = solve(prob, NewtonRaphson())

for i = 1 : N+1
  println(edges[i].vl)
  println(edges[i].vg)
  println(edges[i].mass_flux_l)
  println(edges[i].mass_flux_g)
end
for i = 1 : N
  println(cells[i].alpha)
  println(cells[i].p)
  println(cells[i].rho_l)
  println(cells[i].rho_g)
end
println("*************************")

println(sol)

# # debug:
# for i = 1 : N
#   west_cell = cells[i].cell_west
#   west_edge = cells[i].edge_west
#   east_edge = cells[i].edge_east
#   east_cell = cells[i].cell_east

#   if (west_cell == nothing)
#     println("nothing ->")
#   else
#     println(west_cell.name)
#   end

#   if (west_edge == nothing)
#     println("nothing ->")
#   else
#     println(west_edge.name)
#   end

#   println(cells[i].name)

#   if (east_edge == nothing)
#     println("nothing ->")
#   else
#     println(east_edge.name)
#   end

#   if (east_cell == nothing)
#     println("nothing")
#   else
#     println(east_cell.name)
#   end
#   println("")
# end

# for i = 1 : N+1
#   west_edge = edges[i].edge_west
#   west_cell = edges[i].cell_west
#   east_cell = edges[i].cell_east
#   east_edge = edges[i].edge_east

#   if (west_edge == nothing)
#     println("nothing ->")
#   else
#     println(west_edge.name)
#   end

#   if (west_cell == nothing)
#     println("nothing ->")
#   else
#     println(west_cell.name)
#   end

#   println(edges[i].name)

#   if (east_cell == nothing)
#     println("nothing")
#   else
#     println(east_cell.name)
#   end
#   if (east_edge == nothing)
#     println("nothing")
#   else
#     println(east_edge.name)
#   end
#   println("")
# end