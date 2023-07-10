using LinearAlgebra
using ForwardDiff
using Parameters
using JuMP, HiGHS, ECOS

NumberOrVariable = Union{Number, VariableRef}


compute_quadratic_error_cost(state::Vector{V}, goal::Vector{T}, Qt::Matrix{T}) where {V,T} =  transpose(state - goal) * Qt * (state - goal)

compute_quadratic_cost(state::Vector{V}, Q::Matrix{T}) where {V,T} = compute_quadratic_error_cost(state, zeros(Float64, size(state)), Q)

compute_running_quadratic_cost(states::Vector{V}, Q::Matrix{T}; markup=1.0) where {V,T} = sum([markup^i * compute_quadratic_cost(s, Q) for (i,s) in enumerate(states)])

function compute_total_difference_squared(time_series::Vector{T}) where {T}
    dx = diff(time_series, dims=1)
    n = size(time_series[1])[end]
    Q = Matrix{Float64}(I, n, n)
    compute_running_quadratic_cost(dx, Q)
end    


function compute_convenience_value(dynamics::Dynamics, states::Vector{V}, controls::Vector{V}, goal::Vector{T}, inconvenience_weights::VecOrMat{T}) where {T,V}
    position = get_position(dynamics, states)
    speed = get_speed(dynamics, states, controls)
    n = dynamics.state_dim
    total_squared_distance = compute_total_difference_squared(position)
    total_squared_acceleration = compute_total_difference_squared(speed)
    distance_squared_from_goal = compute_quadratic_error_cost(states[end], goal, Matrix{Float64}(I, n, n))
    v = [total_squared_distance, total_squared_acceleration, distance_squared_from_goal]
    dot(inconvenience_weights, v)
end


function collision_avoidance_constraint(radius::T, ego_dyn::Dynamics, ego_state::Vector{T}, other_dyn::Dynamics, other_state::Vector{T}) where {T<:NumberOrVariable}
    ego_position = get_position(ego_dyn, ego_state)
    other_position = get_position(other_dyn, other_state)
    ego_position - other_position
    norm(ego_position - other_position, 2)^2 - radius^2
end

collision_avoidance_constraint(radius::T, ego_dyn::Dynamics, ego_states::Vector{Vector{T}}, other_dyn::Dynamics, other_states::Vector{Vector{T}}) where {T<:NumberOrVariable} = collision_avoidance_constraint.(Ref(radius), Ref(ego_dyn), ego_states, Ref(other_dyn), other_states)

function collision_avoidance_constraint(radius::T, ego_position::Vector{T}, other_position::Vector{T}) where {T<:NumberOrVariable}
    sum((ego_position - other_position).^2) - radius^2
end
collision_avoidance_constraint(radius::T, ego_positions::Vector{Vector{T}}, other_positions::Vector{Vector{T}}) where {T<:NumberOrVariable} = collision_avoidance_constraint.(Ref(radius), ego_positions, other_positions)


function linearize_collision_avoidance(ego_dyn::Dynamics, ego_state::Vector{T}, other_dyn::Dynamics, other_state::Vector{T}) where {T<:NumberOrVariable}
    ego_position = get_position(ego_dyn, ego_state)
    other_position = get_position(other_dyn, other_state)
    2 * (ego_position - other_position)
end 
linearize_collision_avoidance(ego_dyn::Dynamics, ego_states::Vector{Vector{T}}, other_dyn::Dynamics, other_states::Vector{Vector{T}}) where {T<:NumberOrVariable} = linearize_collision_avoidance.(Ref(ego_dyn), ego_states, Ref(other_dyn), other_states)

function linearize_collision_avoidance(ego_position::Vector, other_position::Vector)
    2 * (ego_position - other_position)
end 

# hyperparameters of a planning problem (both inconvenience and ideal)
@with_kw mutable struct PlannerHyperparameters{T}
    dynamics::Dynamics
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

# initialize hyperparameter values
function PlannerHyperparameters(dyn::Dynamics; time_horizon=20, markup=1.0, collision_slack=1000., trust_region_weight=10., inconvenience_weights=[1. 1. 1.], collision_radius=0.25, inconvenience_ratio=0.1)
    n = dyn.state_dim
    m = dyn.ctrl_dim
    Q = Matrix{Float64}(I, n, n)
    Qt = Matrix{Float64}(I, n, n)
    R = Matrix{Float64}(I, m, m)
    return PlannerHyperparameters(dyn, time_horizon, Q, R, Qt, markup, collision_slack, trust_region_weight, inconvenience_weights, collision_radius, inconvenience_ratio)
end

# parameters (constants that will need to be updated) of an optimization problem (both inconvenience and ideal)
@with_kw mutable struct PlannerOptimizerParams{T}
    As::Vector{Matrix{T}}   # linearized A matrix for dynamics
    Bs::Vector{Matrix{T}}   # linearized B matrix for dynamics
    Cs::Vector{Vector{T}}   # linearized C matrix for dynamics
    Gs::Vector{Matrix{T}}   # linearized collision avoidance constraint linear term wrt position (x,y)
    Hs::Vector{Vector{T}}   # linearized collision avoidance constant term wrt position (x,y)
    inconvenience_budget::T
    initial_state::Vector{T}
    goal_state::Vector{T}
    previous_states::Vector{Vector{T}}
    previous_controls::Vector{Vector{T}}
    solver::String
end

# initialize planner parameters with undef
function PlannerOptimizerParams(dyn::Dynamics, hp::PlannerHyperparameters, solver::String)
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
    previous_states = matrix_to_vector_of_vectors(Matrix{Float64}(undef, hp.time_horizon+1, n))
    previous_controls = matrix_to_vector_of_vectors(Matrix{Float64}(undef, hp.time_horizon, m))
    
    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls, solver)
end

# initialize planner parameters with straight line trajectory
function PlannerOptimizerParams(dyn::Dynamics, hp::PlannerHyperparameters, start_position::Vector{T}, end_position::Vector{T}, solver::String) where {T}
    # initialize planner parameters / allocation space
    n = dyn.state_dim
    m = dyn.ctrl_dim
    N = hp.time_horizon
    p = 2 # size of position dimension
    inconvenience_budget = 1.  # arbitrary number

    # use straight line trajectory
    previous_states_, previous_controls_ = initial_straight_trajectory(dyn, start_position, end_position, dyn.velocity_max * 0.75, hp.time_horizon)
    previous_states = matrix_to_vector_of_vectors(previous_states_)
    previous_controls = matrix_to_vector_of_vectors(previous_controls_)
    initial_state = previous_states[1]
    goal_state = previous_states[end]

    ABCs = linearized_dynamics(dyn, previous_states[1:N], previous_controls[1:N])
    As = [Matrix{Float64}(undef, n, n) for i in 1:hp.time_horizon]
    Bs = [Matrix{Float64}(undef, n, m) for i in 1:hp.time_horizon]
    Cs = [Vector{Float64}(undef, n) for i in 1:hp.time_horizon]
    
    for (t, (A, B, C)) in enumerate(ABCs)
        As[t] = A
        Bs[t] = B
        Cs[t] = C
    end

    # left as undef as the other agent's trajectory is needed
    Gs = [Matrix{Float64}(undef, p, p) for i in 1:hp.time_horizon+1]
    Hs = [Vector{Float64}(undef, p) for i in 1:hp.time_horizon+1]

    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls, solver)
end


# relinearize dynamics with new states and controls
function update_dynamics_linearization!(opt_params::PlannerOptimizerParams, dyn::Dynamics, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where{T}
    N = size(opt_params.As)[1]
    ABCs = linearized_dynamics(dyn, states[1:N], controls[1:N])
    for (t, (A, B, C)) in enumerate(ABCs)
        opt_params.As[t] = A
        opt_params.Bs[t] = B
        opt_params.Cs[t] = C
    end
end

# relinearize dynamics with new states and controls
function update_dynamics_linearization!(opt_params::PlannerOptimizerParams, dyn::Dynamics)
    N = size(opt_params.As)[1]
    states = opt_params.previous_states
    controls = opt_params.previous_controls
    ABCs = linearized_dynamics(dyn, states[1:N], controls[1:N])
    for (t, (A, B, C)) in enumerate(ABCs)
        opt_params.As[t] = A
        opt_params.Bs[t] = B
        opt_params.Cs[t] = C
    end
end


mutable struct InconvenienceProblem{T}
    model::Model
    xs::Vector{Vector{VariableRef}}
    us::Vector{Vector{VariableRef}}
    ϵs::Vector{VariableRef}
    hps::PlannerHyperparameters
    opt_params::PlannerOptimizerParams
    other_states::Vector{Vector{T}}
end

mutable struct IdealProblem
    model::Model
    xs::Vector{Vector{VariableRef}}
    us::Vector{Vector{VariableRef}}
    hps::PlannerHyperparameters
    opt_params::PlannerOptimizerParams
end

# setup ideal problem
function IdealProblem(dyn::Dynamics, hps::PlannerHyperparameters, opt_params::PlannerOptimizerParams)
    n = dyn.state_dim
    m = dyn.ctrl_dim
    N = hps.time_horizon
    radius = hps.collision_radius
    solver = opt_params.solver

    if solver == "ECOS"
        model = Model(ECOS.Optimizer)
    elseif solver == "HiGHS"
        model = Model(HiGHS.Optimizer)
    elseif solver == "Gurobi"
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    end
    
    xs = matrix_to_vector_of_vectors(@variable(model, x[1:N+1,1:dyn.state_dim], base_name="x"))
    us = matrix_to_vector_of_vectors(@variable(model, u[1:N,1:dyn.ctrl_dim], base_name="u"))
    
    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))))

    # initial state constraint
    @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")    
    
    # dynamic constraints
    for t in 1:N
        @constraint(model, opt_params.As[t]*xs[t] + opt_params.Bs[t]*us[t] + opt_params.Cs[t] == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
    end
    
    # control constraints
    for t in 1:N
        @constraint(model, us[t] <= dyn.control_max, base_name="control_constraints_upper_$(t)")
        @constraint(model, dyn.control_min <= us[t] , base_name="control_constraints_lower_$(t)")
        @constraint(model, get_speed(dyn, xs[t], us[t]) .<= dyn.velocity_max , base_name="speed_constraints_upper_$(t)")
        @constraint(model, get_speed(dyn, xs[t], us[t]) .>= dyn.velocity_min , base_name="speed_constraints_lower_$(t)")
    end
    
    IdealProblem(model, xs, us, hps, opt_params)
end
    