module ProactiveHRI

# Write your package code here.
__precompile__()

export Unicycle
export dynamics.jl

include("dynamics.jl")
include("planner.jl")
include("planner_utils.jl")
include("utils.jl")
include("plotting.jl")
include("mpc.jl")
include("sim.jl")

import. SingleIntegrator2D, DoubleIntegrator2D, SingleIntegratorPolar2D, Unicycle, DynamicallyExtendedUnicycle

end
