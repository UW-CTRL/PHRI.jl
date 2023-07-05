using LinearAlgebra

mutable struct PlannerHyperparameters{T}
    dyn::Dynamics
    time_horizon::Int64
    Q::Array{T}
    R::Array{T}
    Qt::Array{T}
    markup::T
    collision_slack::T
    trust_region_weight::T
    inconvenience_weights::VecOrMat{T}
    collision_radius::T
    inconvenience_ratio::T
end

function PlannerHyperparameters(dyn;time_horizon=20, markup=1.0, collision_slack=1000., trust_region_weight=10., inconvenience_weights=[1. 1. 1.], collision_radius=0.25, inconvenience_ratio=0.1)
    n = dyn.state_dim
    m = dyn.ctrl_dim

    Q = Matrix{Float64}(I, n, n)
    Qt = Matrix{Float64}(I, n, n)
    R = Matrix{Float64}(I, m, m)
    PlannerHyperparameters(dyn, time_horizon, Q, R, Qt, markup, collision_slack, trust_region_weight, inconvenience_weights, collision_radius, inconvenience_ratio)
end

mutable struct PlannerOptimizerParams{T}
    As::Vector{Matrix{T}}   # linearized A matrix for dynamics
    Bs::Vector{Matrix{T}}   # linearized B matrix for dynamics
    Cs::Vector{Vector{T}}   # linearized C matrix for dynamics
    Gs::Vector{Matrix{T}}   # linearized collision avoidance constraint linear term wrt position (x,y)
    Hs::Vector{Vector{T}}   # linearized collision avoidance constant term wrt position (x,y)
    inconvenience_budget::T
    initial_state::Vector{T}
    goal_state::Vector{T}
    previous_states::Matrix{T}
    previous_controls::Matrix{T}
end

function InitializePlannerOptimizerParams(dyn::Dynamics, hp::PlannerHyperparameters)
    # initialize planner parameters / allocation space
    n = dyn.state_dim
    m = dyn.ctrl_dim
    p = 2 # size of position dimension
    As = [Matrix{Float64}(undef, n, n) for i in 1:hp.time_horizon]
    Bs = [Matrix{Float64}(undef, n, m) for i in 1:hp.time_horizon]
    Cs = [Vector{Float64}(undef, n) for i in 1:hp.time_horizon]
    Gs = [Matrix{Float64}(undef, p, p) for i in 1:hp.time_horizon+1]
    Hs = [Vector{Float64}(undef, p) for i in 1:hp.time_horizon+1]

    inconvenience_budget = 1.
    initial_state = zeros(Float64, n)
    goal_state = ones(Float64, n)
    previous_states = Matrix{Float64}(undef, hp.time_horizon+1, n)
    previous_controls = Matrix{Float64}(undef, hp.time_horizon, m)
    
    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls)
end

# function update_planner_params!(planner_params, prev_state, prev_control, initial_state, goal_state)
#     ideal_traj = ....
#     ...
# end


