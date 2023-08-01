module ProactiveHRI

# Write your package code here.
using LinearAlgebra
using ForwardDiff
using Parameters
using JuMP, HiGHS, ECOS
using EllipsisNotation
using Parameters
using AngleBetweenVectors

import LinearAlgebra: diagm

export diagm
export SingleIntegrator2D, DoubleIntegrator2D, SingleIntegratorPolar2D, Unicycle, DynamicallyExtendedUnicycle
export PlannerHyperparameters, PlannerOptimizerParams, InconvenienceProblem, IdealProblem, update_problem!, solve, InteractionPlanner, ibr, ibr_save, ibr_mpc
export mpc_step, Sim
export plot_solve_solution, animation, avoidance_animation

include("dynamics.jl")
include("planner.jl")
include("planner_utils.jl")
include("utils.jl")
include("plotting.jl")
include("mpc.jl")
include("sim.jl")

end
