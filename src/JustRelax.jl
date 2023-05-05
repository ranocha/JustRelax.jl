module JustRelax

using Reexport
@reexport using ParallelStencil
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using MPI
using GeoParams
using HDF5
using CellArrays
using StaticArrays

function solve!() end

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry

include("MetaJustRelax.jl")

include("stokes/MetaStokes.jl")
export PS_Setup, environment!, ps_reset!

include("thermal_diffusion/MetaDiffusion.jl")

# include("phases/phases.jl")
export Phases, PhaseRatio, nphases, phase_ratios_center

include("IO/DataIO.jl")

end # module
