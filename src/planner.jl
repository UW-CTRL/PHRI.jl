using LinearAlgebra
using JuMP, HiGHS, ECOS

abstract type Dynamics end
abstract type Parameters end

mutable struct PlannerHyperparameters{T} <: Parameters
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

function PlannerHyperparameters(dyn::Dynamics, time_horizon=20, markup=1.0, collision_slack=1000., trust_region_weight=10., inconvenience_weights=[1. 1. 1.], collision_radius=0.25, inconvenience_ratio=0.1)
    n = dyn.state_dim
    m = dyn.ctrl_dim

    Q = Matrix{Float64}(I, n, n)
    Qt = Matrix{Float64}(I, n, n)
    R = Matrix{Float64}(I, m, m)
    return PlannerHyperparameters(dyn, time_horizon, Q, R, Qt, markup, collision_slack, trust_region_weight, inconvenience_weights, collision_radius, inconvenience_ratio)
end

mutable struct PlannerOptimizerParams{T} <: Parameters
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
    flag::String
end

function InitializePlannerOptimizerParams(dyn::Dynamics, hp::PlannerHyperparameters, flag="highs")
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
    
    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls, flag)
end

# function update_planner_params!(Plan, prev_state, prev_control, initial_state, goal_state)
#     ideal_traj = ...
# end

mutable struct ModelInitialization
    model::Model
    x::Array{VariableRef, 2}
    u::Array{VariableRef, 2}
    slack::Array{VariableRef, 1}
end


function initialializeInconvenienceProblem(dyn::Dynamics, hp::Parameters, op::Parameters)
    local markup = hp.markup
    local slack_weight = hp.collision_slack
    local Q = hp.Q
    local Qt = hp.Qt
    local R = hp.R
    local N = hp.time_horizon
    local statef = op.goal_state

    flag = op.flag

    if flag == "ecos"
        model = Model(ECOS.Optimizer)
    elseif flag == "highs"
        model = Model(HiGHS.Optimizer)
    else
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    end

    @variable(model, x[ 1:N + 1, 1:dyn.state_dim])      # initialize state variable for qp_model
    @variable(model, u[1:N, 1:dyn.ctrl_dim])           # initialize control variable for qp_model
    @variable(model, slack[1:N])                       # initialize slack variable for qp_model

    @objective(
        model,
        Min,
        sum(x[n, :]' * Q * x[n, :] for n in 1:N) + sum(u[n, :]' * R * u[n, :] * markup^n for n in 1:N) + (x[N + 1, :] - statef)' * Qt * (x[N + 1, :] - statef) + sum(slack[n] * slack_weight for n in 1:N)
    )   

    @constraint(model, dyn <= get_velocity(dyn, x[n, :], u[n, :] <= v_max for n = 1:N))
    @constraint(model, u[:, n] <= u_max for n = 1:N)

    return ModelInitialization(model, x, u, slack)
end

function InitializeIdealProblem()   # TODO
end