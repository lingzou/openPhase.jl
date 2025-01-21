using BenchmarkTools

module TF4E1P

# using MacroTools

export Cell, initialize!, updateSolution!, saveOldSolutions!, makeConnection!, extendConnection!, computeFluxes!
export IntEdge, pBndryEdge, vBndryEdge
export assignDOF!
export mass_eqn, momentum_eqn, mass_eqn!, momentum_eqn!
export printName

# export AbstractEdge, AbstractCell

# fluid properties
rho_l_fn(p) = 1e3 + 1e-7 * (p - 1e5)
rho_g_fn(p) = 0.5 + 1e-5 * (p - 1e5)

abstract type AbstractEdge end
abstract type AbstractCell end

# debug
function printName(obj::Union{AbstractEdge, AbstractCell, Nothing})
  if (obj == nothing)
    println("nothing")
  else
    println(obj.name)
  end
end

"""
  Connections
"""
getOtherSideCell(edge::AbstractEdge, cell::Union{AbstractCell, Nothing}) =
  (edge == nothing) ? nothing : ((cell == edge.w_cell) ? edge.e_cell : edge.w_cell)
getOtherSideEdge(edge::AbstractEdge, cell::Union{AbstractCell, Nothing}) =
  (cell == nothing) ? nothing : ((edge == cell.w_edge) ? cell.e_edge : cell.w_edge)

function extendConnection!(cell::AbstractCell)
  cell.w_cell = getOtherSideCell(cell.w_edge, cell)
  cell.e_cell = getOtherSideCell(cell.e_edge, cell)
end
function extendConnection!(edge::AbstractEdge)
  edge.w_edge = getOtherSideEdge(edge, edge.w_cell)
  edge.e_edge = getOtherSideEdge(edge, edge.e_cell)
end

"""
  A one-dimensional finite volume cell
"""
Base.@kwdef mutable struct Cell <: AbstractCell
  name        ::String
  dx          ::Float64
  alpha_dof   ::UInt32  = 0
  p_dof       ::UInt32  = 0
  alpha       = 0.0 #::Float64
  p           = 1.0e5 #::Float64
  rho_l       = rho_l_fn(1.0e5)
  rho_g       = rho_g_fn(1.0e5)
  alpha_o     = alpha
  rho_l_o     = rho_l
  rho_g_o     = rho_g
  alpha_oo    = alpha
  rho_l_oo    = rho_l
  rho_g_oo    = rho_g
  w_edge      ::Union{AbstractEdge, Nothing}  = nothing
  e_edge      ::Union{AbstractEdge, Nothing}  = nothing
  w_cell      ::Union{AbstractCell, Nothing}  = nothing
  e_cell      ::Union{AbstractCell, Nothing}  = nothing
end

function initialize!(cell::Cell, alpha_init, p_init)
  cell.alpha, cell.alpha_o, cell.alpha_oo = alpha_init, alpha_init, alpha_init
  cell.p = p_init
  rho_l = rho_l_fn(p_init)
  rho_g = rho_g_fn(p_init)
  cell.rho_l, cell.rho_l_o, cell.rho_l_oo = rho_l, rho_l, rho_l
  cell.rho_g, cell.rho_g_o, cell.rho_g_oo = rho_g, rho_g, rho_g
end

function updateSolution!(cell::Cell, alpha, p)
  cell.alpha, cell.p = alpha, p
  cell.rho_l, cell.rho_g = rho_l_fn(p), rho_g_fn(p)
end

function updateSolution!(cell::Cell, u)
  cell.alpha, cell.p = u[cell.alpha_dof], u[cell.p_dof]
  cell.rho_l, cell.rho_g = rho_l_fn(cell.p), rho_g_fn(cell.p)
end

function assignDOF!(cell::Cell, alpha_dof, p_dof)
  cell.alpha_dof, cell.p_dof = alpha_dof, p_dof
end

function saveOldSolutions!(cell::Cell)
  # old solutions -> oldold solutions
  # solutions -> old solutions
  cell.alpha_oo = cell.alpha_o
  cell.alpha_o = cell.alpha

  cell.rho_l_oo, cell.rho_g_oo = cell.rho_l_o, cell.rho_g_o
  cell.rho_l_o, cell.rho_g_o = cell.rho_l, cell.rho_g
end

function makeConnection!(west_edge::AbstractEdge, cell::Cell, east_edge::AbstractEdge)
  cell.w_edge, cell.e_edge = west_edge, east_edge
  west_edge.e_cell, east_edge.w_cell = cell, cell
end

function mass_eqn(cell::Cell, dt, dx)
  res_l = ((1.0 - cell.alpha) * cell.rho_l - (1.0 - cell.alpha_o) * cell.rho_l_o) / dt
  res_g = (cell.alpha * cell.rho_g - cell.alpha_o * cell.rho_g_o) / dt

  res_l += (cell.e_edge.mass_flux_l - cell.w_edge.mass_flux_l) / dx
  res_g += (cell.e_edge.mass_flux_g - cell.w_edge.mass_flux_g) / dx

  res_l, res_g
end

function mass_eqn!(cell::Cell, dt, dx, res)
  res[1] = ((1.0 - cell.alpha) * cell.rho_l - (1.0 - cell.alpha_o) * cell.rho_l_o) / dt
  res[2] = (cell.alpha * cell.rho_g - cell.alpha_o * cell.rho_g_o) / dt

  res[1] += (cell.e_edge.mass_flux_l - cell.w_edge.mass_flux_l) / dx
  res[2] += (cell.e_edge.mass_flux_g - cell.w_edge.mass_flux_g) / dx
end


"""
  A internal edge that connects two one-dimensional cells (Cell)
"""
Base.@kwdef mutable struct IntEdge <: AbstractEdge
  name        ::String
  vl_dof      ::UInt32  = 0
  vg_dof      ::UInt32  = 0
  vl          = 0.0 #::Float64
  vg          = 0.0 #::Float64
  vl_o        = vl
  vg_o        = vl
  vl_oo       = vl
  vg_oo       = vl
  mass_flux_l = 0.0
  mass_flux_g = 0.0
  rho_l_avg   = 0.0
  rho_g_avg   = 0.0
  w_cell      ::Union{AbstractCell, Nothing}  = nothing
  e_cell      ::Union{AbstractCell, Nothing}  = nothing
  w_edge      ::Union{AbstractEdge, Nothing}  = nothing
  e_edge      ::Union{AbstractEdge, Nothing}  = nothing
end

function initialize!(edge::IntEdge, vl_init, vg_init)
  edge.vl, edge.vl_o, edge.vl_oo = vl_init, vl_init, vl_init
  edge.vg, edge.vg_o, edge.vg_oo = vg_init, vg_init, vg_init
end

function assignDOF!(edge::IntEdge, vl_dof, vg_dof)
  edge.vl_dof, edge.vg_dof = vl_dof, vg_dof
end

function updateSolution!(edge::IntEdge, vl, vg)
  edge.vl, edge.vg = vl, vg
end

function saveOldSolutions!(edge::IntEdge)
  # old solutions -> oldold solutions
  # solutions -> old solutions
  edge.vl_oo, edge.vg_oo = edge.vl_o, edge.vg_o
  edge.vl_o, edge.vg_o = edge.vl, edge.vg
end

function computeFluxes!(edge::IntEdge)
  edge.rho_l_avg = 0.5 * (edge.w_cell.rho_l + edge.e_cell.rho_l)
  edge.rho_g_avg = 0.5 * (edge.w_cell.rho_g + edge.e_cell.rho_g)

  edge.mass_flux_l = ifelse(edge.vl > 0.0,
                            edge.vl * edge.w_cell.rho_l * (1.0 - edge.w_cell.alpha),
                            edge.vl * edge.e_cell.rho_l * (1.0 - edge.e_cell.alpha))
#
  edge.mass_flux_g = ifelse(edge.vg > 0.0,
                            edge.vg * edge.w_cell.rho_g * edge.w_cell.alpha,
                            edge.vg * edge.e_cell.rho_g * edge.e_cell.alpha)
end

function momentum_eqn(edge::IntEdge, dt, dx)
  res_l = (edge.vl - edge.vl_o) / dt
  res_g = (edge.vg - edge.vg_o) / dt

  adv_l = edge.vl / dx * ifelse(edge.vl > 0.0, edge.vl - edge.w_edge.vl, edge.e_edge.vl - edge.vl)
  adv_g = edge.vg / dx * ifelse(edge.vg > 0.0, edge.vg - edge.w_edge.vg, edge.e_edge.vg - edge.vg)
  dp_dx = (edge.e_cell.p - edge.w_cell.p) / dx

  res_l += (adv_l + dp_dx / edge.rho_l_avg - 9.8)
  res_g += (adv_g + dp_dx / edge.rho_g_avg - 9.8)

  res_l, res_g
end

function momentum_eqn!(edge::IntEdge, dt, dx, res)
  res[1] = (edge.vl - edge.vl_o) / dt
  res[2] = (edge.vg - edge.vg_o) / dt

  adv_l = edge.vl / dx * ifelse(edge.vl > 0.0, edge.vl - edge.w_edge.vl, edge.e_edge.vl - edge.vl)
  adv_g = edge.vg / dx * ifelse(edge.vg > 0.0, edge.vg - edge.w_edge.vg, edge.e_edge.vg - edge.vg)
  dp_dx = (edge.e_cell.p - edge.w_cell.p) / dx

  res[1] += (adv_l + dp_dx / edge.rho_l_avg - 9.8)
  res[2] += (adv_g + dp_dx / edge.rho_g_avg - 9.8)
end

"""
  pressure boundary edge
"""
Base.@kwdef mutable struct pBndryEdge <: AbstractEdge
  name        ::String
  vl_dof      ::UInt32  = 0
  vg_dof      ::UInt32  = 0
  p_bc
  alpha_bc
  vl          = 0.0 #::Float64
  vg          = 0.0 #::Float64
  vl_o        = vl
  vg_o        = vl
  vl_oo       = vl
  vg_oo       = vl
  mass_flux_l = 0.0
  mass_flux_g = 0.0
  rho_l_avg   = 0.0
  rho_g_avg   = 0.0
  w_cell      ::Union{AbstractCell, Nothing}  = nothing
  e_cell      ::Union{AbstractCell, Nothing}  = nothing
  w_edge      ::Union{AbstractEdge, Nothing}  = nothing
  e_edge      ::Union{AbstractEdge, Nothing}  = nothing
end

function initialize!(edge::pBndryEdge, vl_init, vg_init)
  edge.vl, edge.vl_o, edge.vl_oo = vl_init, vl_init, vl_init
  edge.vg, edge.vg_o, edge.vg_oo = vg_init, vg_init, vg_init
end

function updateSolution!(edge::pBndryEdge, vl, vg)
  edge.vl, edge.vg = vl, vg
end

function assignDOF!(edge::pBndryEdge, vl_dof, vg_dof)
  edge.vl_dof, edge.vg_dof = vl_dof, vg_dof
end

function saveOldSolutions!(edge::pBndryEdge)
  # old solutions -> oldold solutions
  # solutions -> old solutions
  edge.vl_oo, edge.vg_oo = edge.vl_o, edge.vg_o
  edge.vl_o, edge.vg_o = edge.vl, edge.vg
end

function computeFluxes!(edge::pBndryEdge)
  if (edge.w_cell == nothing)
    edge.rho_l_avg, edge.rho_g_avg = edge.e_cell.rho_l, edge.e_cell.rho_g

    edge.mass_flux_l = ifelse(edge.vl > 0.0,
                              edge.vl * rho_l_fn(edge.p_bc) * (1.0 - edge.alpha_bc),
                              edge.vl * edge.e_cell.rho_l * (1.0 - edge.e_cell.alpha))
    edge.mass_flux_g = ifelse(edge.vg > 0.0,
                              edge.vg * rho_g_fn(edge.p_bc) * edge.alpha_bc,
                              edge.vg * edge.e_cell.rho_g * edge.e_cell.alpha)
  else
    edge.rho_l_avg, edge.rho_g_avg = edge.w_cell.rho_l, edge.w_cell.rho_g

    edge.mass_flux_l = ifelse(edge.vl > 0.0,
                              edge.vl * edge.w_cell.rho_l * (1.0 - edge.w_cell.alpha),
                              edge.vl * rho_l_fn(edge.p_bc) * (1.0 - edge.alpha_bc))
    edge.mass_flux_g = ifelse(edge.vg > 0.0,
                              edge.vg * edge.w_cell.rho_g * edge.w_cell.alpha,
                              edge.vg * rho_g_fn(edge.p_bc) * edge.alpha_bc)
  end
end

function momentum_eqn(edge::pBndryEdge, dt, dx)
  res_l = (edge.vl - edge.vl_o) / dt
  res_g = (edge.vg - edge.vg_o) / dt

  if (edge.w_cell == nothing)
    adv_l = edge.vl / dx * ifelse(edge.vl > 0.0, 0.0, edge.e_edge.vl - edge.vl)
    adv_g = edge.vg / dx * ifelse(edge.vg > 0.0, 0.0, edge.e_edge.vg - edge.vg)
    dp_dx = (edge.e_cell.p - edge.p_bc) / dx * 2.0
  else
    adv_l = edge.vl / dx * ifelse(edge.vl > 0.0, edge.vl - edge.w_edge.vl, 0.0)
    adv_g = edge.vg / dx * ifelse(edge.vg > 0.0, edge.vg - edge.w_edge.vg, 0.0)
    dp_dx = (edge.p_bc - edge.w_cell.p) / dx * 2.0
  end

  res_l += (adv_l + dp_dx / edge.rho_l_avg - 9.8)
  res_g += (adv_g + dp_dx / edge.rho_g_avg - 9.8)

  res_l, res_g
end

function momentum_eqn!(edge::pBndryEdge, dt, dx, res)
  res[1] = (edge.vl - edge.vl_o) / dt
  res[2] = (edge.vg - edge.vg_o) / dt

  if (edge.w_cell == nothing)
    adv_l = edge.vl / dx * ifelse(edge.vl > 0.0, 0.0, edge.e_edge.vl - edge.vl)
    adv_g = edge.vg / dx * ifelse(edge.vg > 0.0, 0.0, edge.e_edge.vg - edge.vg)
    dp_dx = (edge.e_cell.p - edge.p_bc) / dx * 2.0
  else
    adv_l = edge.vl / dx * ifelse(edge.vl > 0.0, edge.vl - edge.w_edge.vl, 0.0)
    adv_g = edge.vg / dx * ifelse(edge.vg > 0.0, edge.vg - edge.w_edge.vg, 0.0)
    dp_dx = (edge.p_bc - edge.w_cell.p) / dx * 2.0
  end

  res[1] += (adv_l + dp_dx / edge.rho_l_avg - 9.8)
  res[2] += (adv_g + dp_dx / edge.rho_g_avg - 9.8)
end

"""
  velocity boundary edge
"""
Base.@kwdef mutable struct vBndryEdge <: AbstractEdge
  name        ::String
  vl_dof      ::UInt32  = 0
  vg_dof      ::UInt32  = 0
  vl_bc
  vg_bc
  alpha_bc
  vl          = 0.0
  vg          = 0.0
  mass_flux_l = 0.0
  mass_flux_g = 0.0
  rho_l_avg   = 0.0
  rho_g_avg   = 0.0
  w_cell      ::Union{AbstractCell, Nothing}  = nothing
  e_cell      ::Union{AbstractCell, Nothing}  = nothing
  w_edge      ::Union{AbstractEdge, Nothing}  = nothing
  e_edge      ::Union{AbstractEdge, Nothing}  = nothing
end

function initialize!(edge::vBndryEdge, vl_bc, vg_bc)
  edge.vl, edge.vl_bc = vl_bc, vl_bc
  edge.vg, edge.vg_bc = vg_bc, vg_bc
end

function updateSolution!(edge::vBndryEdge, vl, vg)
  edge.vl, edge.vg = vl, vg
end

function assignDOF!(edge::vBndryEdge, vl_dof, vg_dof)
  edge.vl_dof, edge.vg_dof = vl_dof, vg_dof
end

function computeFluxes!(edge::vBndryEdge)
  if (edge.w_cell == nothing)
    edge.rho_l_avg, edge.rho_g_avg = edge.e_cell.rho_l, edge.e_cell.rho_g

    # vBndryEdge has no p_bc, use 0-th order projection
    rho_l_ghost, rho_g_ghost = rho_l_fn(edge.e_cell.p), rho_g_fn(edge.e_cell.p)
    edge.mass_flux_l = ifelse(edge.vl > 0.0,
                              edge.vl * rho_l_ghost * (1.0 - edge.alpha_bc),
                              edge.vl * rho_g_ghost * (1.0 - edge.e_cell.alpha))
    edge.mass_flux_g = ifelse(edge.vg > 0.0,
                              edge.vg * rho_g_ghost * edge.alpha_bc,
                              edge.vg * edge.e_cell.rho_g * edge.e_cell.alpha)
  else
    edge.rho_l_avg, edge.rho_g_avg = edge.w_cell.rho_l, edge.w_cell.rho_g

    # vBndryEdge has no p_bc, use 0-th order projection
    rho_l_ghost, rho_g_ghost = rho_l_fn(edge.w_cell.p), rho_g_fn(edge.w_cell.p)
    edge.mass_flux_l = ifelse(edge.vl > 0.0,
                              edge.vl * edge.w_cell.rho_l * (1.0 - edge.w_cell.alpha),
                              edge.vl * rho_l_ghost * (1.0 - edge.alpha_bc))
    edge.mass_flux_g = ifelse(edge.vg > 0.0,
                              edge.vg * edge.w_cell.rho_g * edge.w_cell.alpha,
                              edge.vg * rho_g_ghost * edge.alpha_bc)
  end
end

function momentum_eqn(edge::vBndryEdge, dt, dx)
  edge.vl - edge.vl_bc, edge.vg - edge.vg_bc
end

function momentum_eqn!(edge::vBndryEdge, dt, dx, res)
  res[1], res[2] = edge.vl - edge.vl_bc, edge.vg - edge.vg_bc
end

end