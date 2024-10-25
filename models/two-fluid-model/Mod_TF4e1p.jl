module TF4E1P

# fluid properties
rho_l_fn(p) = 1e3 + 1e-7 * (p - 1e5)
rho_g_fn(p) = 0.5 + 1e-5 * (p - 1e5)

abstract type AbstractEdge4E1P end
abstract type AbstractCell4E1P end

export westBndryCell, eastBndryCell, Cell4E1P, Edge4E1P, connect_edge!, extendEdgeConnection!, computeFluxes!, mass_eqn!

Base.@kwdef mutable struct Cell4E1P <: AbstractCell4E1P
  name        ::String
  dx          ::Float64
  alpha       = 0.0 #::Float64
  p           = 1.0e5 #::Float64
  rho_l       = rho_l_fn(1.0e5)
  rho_g       = rho_g_fn(1.0e5)
  alpha_o     = alpha
  rho_l_o     = rho_l
  rho_g_o     = rho_g
  edge_west   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  edge_east   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  cell_west   ::Union{AbstractCell4E1P, Nothing}  = nothing
  cell_east   ::Union{AbstractCell4E1P, Nothing}  = nothing
end
# Cell4E1P(name, dx) = Cell4E1P(name, dx, 0.0, 0.0, 0.0, 0.0, nothing, nothing, nothing, nothing)
# Cell4E1P(name, dx, alpha, p) = Cell4E1P(name, dx, alpha, p, 0.0, 0.0, nothing, nothing, nothing, nothing)

Base.@kwdef mutable struct westBndryCell <: AbstractCell4E1P
  name        ::String
  alpha       = 0.0
  p           = 1.0e5
  rho_l       = rho_l_fn(1.0e5)
  rho_g       = rho_g_fn(1.0e5)
  # rho_l_o     = rho_l
  # rho_g_o     = rho_g
  edge_west   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  edge_east   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  cell_west   ::Union{AbstractCell4E1P, Nothing}  = nothing
  cell_east   ::Union{AbstractCell4E1P, Nothing}  = nothing
end
westBndryCell(name) = westBndryCell(name, 0.0, 0.0, 0.0, 0.0, nothing, nothing, nothing, nothing)
westBndryCell(name, alpha, p) = westBndryCell(name, alpha, p, 0.0, 0.0, nothing, nothing, nothing, nothing)

Base.@kwdef mutable struct eastBndryCell <: AbstractCell4E1P
  name        ::String
  alpha       = 0.0
  p           = 1.0e5
  rho_l       = rho_l_fn(1.0e5)
  rho_g       = rho_g_fn(1.0e5)
  # rho_l_o     = rho_l
  # rho_g_o     = rho_g
  edge_west   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  edge_east   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  cell_west   ::Union{AbstractCell4E1P, Nothing}  = nothing
  cell_east   ::Union{AbstractCell4E1P, Nothing}  = nothing
end
eastBndryCell(name) = eastBndryCell(name, 0.0, 0.0, 0.0, 0.0, nothing, nothing, nothing, nothing)
eastBndryCell(name, alpha, p) = eastBndryCell(name, alpha, p, 0.0, 0.0, nothing, nothing, nothing, nothing)

Base.@kwdef mutable struct Edge4E1P <: AbstractEdge4E1P
  name        ::String
  vl          = 0.0 #::Float64
  vg          = 0.0 #::Float64
  mass_flux_l = 0.0
  mass_flux_g = 0.0
  rho_l_avg   = 0.0
  rho_g_avg   = 0.0
  cell_west   ::Union{AbstractCell4E1P, Nothing}  = nothing
  cell_east   ::Union{AbstractCell4E1P, Nothing}  = nothing
  edge_west   ::Union{AbstractEdge4E1P, Nothing}  = nothing
  edge_east   ::Union{AbstractEdge4E1P, Nothing}  = nothing
end
Edge4E1P(name) = Edge4E1P(name, 0.0, 0.0, 0.0, 0.0, nothing, nothing, nothing, nothing)
Edge4E1P(name, vl, vg) = Edge4E1P(name, vl, vg, 0.0, 0.0, nothing, nothing, nothing, nothing)

function computeFluxes!(edge::Edge4E1P)
  edge.rho_l_avg = 0.5 * (edge.cell_west.rho_l + edge.cell_east.rho_l)
  edge.rho_g_avg = 0.5 * (edge.cell_west.rho_g + edge.cell_east.rho_g)

  edge.mass_flux_l = ifelse(edge.vl > 0.0,
                              edge.vl * edge.cell_west.rho_l * (1.0 - edge.cell_west.alpha),
                              edge.vl * edge.cell_east.rho_l * (1.0 - edge.cell_east.alpha))
  #
  edge.mass_flux_g = ifelse(edge.vg > 0.0,
                              edge.vg * edge.cell_west.rho_g * edge.cell_west.alpha,
                              edge.vg * edge.cell_east.rho_g * edge.cell_east.alpha)
end

function mass_eqn!(res_g, res_l, cell::Cell4E1P, dt, dx)
  res_l = ((1.0 - cell.alpha) * cell.rho_l - (1.0 - cell.alpha_o) * cell.rho_l_o) / dt;
  res_g = (       cell.alpha  * cell.rho_g -        cell.alpha_o  * cell.rho_g_o) / dt;
  res_l += (cell.edge_east.mass_flux_l - cell.edge_west.mass_flux_l)
  res_g += (cell.edge_east.mass_flux_g - cell.edge_west.mass_flux_g)
end

function momentum_eqn!(res_l, res_g, edge::Edge4E1P, dt, dx)
  dvl_dx = (edge.vl > 0.0) ? (edge.vl - edge.edge_west.vl) / dx : (edge.edge_east.vl - edge.vl) / dx
  dvg_dx = (edge.vg > 0.0) ? (edge.vg - edge.edge_west.vg) / dx : (edge.edge_east.vg - edge.vg) / dx
end

function connect_edge!(cell_w::Union{AbstractCell4E1P, Nothing},
                        edge::Edge4E1P,
                        cell_e::Union{AbstractCell4E1P, Nothing})
  edge.cell_west = cell_w
  edge.cell_east = cell_e
  if (cell_w != nothing)
    cell_w.edge_east = edge
    cell_w.cell_east = cell_e
  end
  if (cell_e != nothing)
    cell_e.edge_west = edge
    cell_e.cell_west = cell_w
  end
end

function extendEdgeConnection!(edge::AbstractEdge4E1P)
  if (edge.cell_west != nothing)
    edge.edge_west = getOtherSideEdge(edge, edge.cell_west)
  end
  if (edge.cell_east != nothing)
    edge.edge_east = getOtherSideEdge(edge, edge.cell_east)
  end
end

function getOtherSideEdge(edge::AbstractEdge4E1P, cell::AbstractCell4E1P)
  if (cell == nothing)
    return nothing
  else
    if (edge == cell.edge_west)
      return cell.edge_east
    else
      return cell.edge_west
    end
  end
end

# export Cell4E1P
# export Edge4E1P
# export connect

end