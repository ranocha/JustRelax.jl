module JustRelax

using Reexport
@reexport using ParallelStencil
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using MPI

include("topology/Topology.jl")
include("MetaJustRelax.jl")
include("stokes/MetaStokes.jl")
include("thermal_diffusion/MetaDiffusion.jl")
include("TwoPhaseFlow/MetaTPF.jl")

function solve!() end

export PS_Setup, environment!, ps_reset!

end # module
