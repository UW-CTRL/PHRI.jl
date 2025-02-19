using LinearAlgebra
using ForwardDiff
using Parameters
using JuMP, HiGHS, ECOS

NumberOrVariable = Union{Number, VariableRef}

mutable struct ConstantVeloAgent
    pos::Vector{Float64}
    velo::Vector{Float64}
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
    collision_markup::T=0.98
    trust_region_weight::T
    inconvenience_weights::VecOrMat{T}
    collision_radius::T
    inconvenience_ratio::T
end

# initialize hyperparameter values
function PlannerHyperparameters(dyn::Dynamics; time_horizon=20, markup=1.0, collision_slack=1000., collision_markup=0.98::Float64, trust_region_weight=10., inconvenience_weights=[1. 1. 1.], collision_radius=0.25, inconvenience_ratio=0.1)
    n = dyn.state_dim
    m = dyn.ctrl_dim
    Q = Matrix{Float64}(I, n, n)
    Qt = Matrix{Float64}(I, n, n)
    R = Matrix{Float64}(I, m, m)
    return PlannerHyperparameters(dyn, time_horizon, Q, R, Qt, markup, collision_slack, collision_markup, trust_region_weight, inconvenience_weights, collision_radius, inconvenience_ratio)
end

# parameters (constants that will need to be updated) of an optimization problem (both inconvenience and ideal)
@with_kw mutable struct PlannerOptimizerParams{T}
    As::Vector{Matrix{T}}   # linearized A matrix for dynamics
    Bs::Vector{Matrix{T}}   # linearized B matrix for dynamics
    Cs::Vector{Vector{T}}   # linearized C matrix for dynamics
    Gs::Vector{Vector{T}}   # linearized collision avoidance constraint linear term wrt position (x,y)
    Hs::Vector{T}   # linearized collision avoidance constant term wrt position (x,y)
    inconvenience_budget::T
    initial_state::Vector{T}
    goal_state::Vector{T}
    previous_states::Vector{Vector{T}}
    previous_controls::Vector{Vector{T}}
    other_positions::Vector{Vector{T}}
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
    Gs = [Vector{Float64}(undef, p) for i in 1:hp.time_horizon+1]
    Hs = Vector{Float64}(undef, hp.time_horizon+1)

    inconvenience_budget = 1.
    initial_state = zeros(Float64, n)
    goal_state = ones(Float64, n)
    previous_states = matrix_to_vector_of_vectors(Matrix{Float64}(undef, hp.time_horizon+1, n))
    previous_controls = matrix_to_vector_of_vectors(Matrix{Float64}(undef, hp.time_horizon, m))
    other_positions = [Vector{Float64}(undef, p) for i in 1:hp.time_horizon+1]

    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls, other_positions, solver)
end

# initialize planner parameters with straight line trajectory
function PlannerOptimizerParams(dyn::Dynamics, hp::PlannerHyperparameters, start_state::Vector{T}, end_state::Vector{T}, solver::String) where {T}
    # initialize planner parameters / allocation space
    n = dyn.state_dim
    m = dyn.ctrl_dim
    N = hp.time_horizon
    p = 2 # size of position dimension
    inconvenience_budget = 1.  # arbitrary number
    start_position = get_position(dyn, start_state)
    end_position = get_position(dyn, end_state)
    # use straight line trajectory
    previous_states_, previous_controls_ = initial_straight_trajectory(dyn, start_position, end_position, dyn.velocity_max * 0.75, hp.time_horizon)
    previous_states = matrix_to_vector_of_vectors(previous_states_)
    previous_controls = matrix_to_vector_of_vectors(previous_controls_)
    initial_state = start_state
    goal_state = end_state

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
    Gs = [Vector{Float64}(undef, p) for i in 1:hp.time_horizon+1]
    Hs = Vector{Float64}(undef, hp.time_horizon+1)
    other_positions = [Vector{Float64}(undef, p) for i in 1:hp.time_horizon+1]

    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls, other_positions, solver)
end

function PlannerOptimizerParams(dyn::Dynamics, hps::PlannerHyperparameters, start_state::Vector{T}, end_state::Vector{T}, other_positions::Vector{Vector{T}}, solver::String) where {T}
    # initialize planner parameters / allocation space
    n = dyn.state_dim
    m = dyn.ctrl_dim
    N = hps.time_horizon
    p = 2 # size of position dimension
    inconvenience_budget = 1.  # arbitrary number
    start_position = get_position(dyn, start_state)
    end_position = get_position(dyn, end_state)
    # use straight line trajectory
    previous_states_, previous_controls_ = initial_straight_trajectory(dyn, start_position, end_position, dyn.velocity_max * 0.75, hps.time_horizon)
    previous_states = matrix_to_vector_of_vectors(previous_states_)
    previous_controls = matrix_to_vector_of_vectors(previous_controls_)
    initial_state = start_state
    goal_state = end_states

    ABCs = linearized_dynamics(dyn, previous_states[1:N], previous_controls[1:N])
    As = [Matrix{Float64}(undef, n, n) for i in 1:hps.time_horizon]
    Bs = [Matrix{Float64}(undef, n, m) for i in 1:hps.time_horizon]
    Cs = [Vector{Float64}(undef, n) for i in 1:hps.time_horizon]

    for (t, (A, B, C)) in enumerate(ABCs)
        As[t] = A
        Bs[t] = B
        Cs[t] = C
    end

    ps = get_position(robot, previous_states)
    Gs = linearize_collision_avoidance(ps, other_positions)
    Hs = collision_avoidance_constraint(hps.collision_radius, ps, other_positions) - dot.(Gs, ps)

    return PlannerOptimizerParams(As, Bs, Cs, Gs, Hs, inconvenience_budget, initial_state, goal_state, previous_states, previous_controls, other_positions, solver)
end

abstract type Problem end
abstract type Planner end

mutable struct InconvenienceProblem <: Problem
    model::Model
    xs::Vector{Vector{VariableRef}}
    us::Vector{Vector{VariableRef}}
    ϵ::Vector{VariableRef}
    hps::PlannerHyperparameters
    opt_params::PlannerOptimizerParams
end

mutable struct IdealProblem <: Problem
    model::Model
    xs::Vector{Vector{VariableRef}}
    us::Vector{Vector{VariableRef}}
    hps::PlannerHyperparameters
    opt_params::PlannerOptimizerParams
end

# setup inconvenience problem
function InconvenienceProblem(dyn::Union{UnicycleDynamics, SingleIntegratorPolar2D}, hps::PlannerHyperparameters, opt_params::PlannerOptimizerParams)
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
    model[:x] = @variable(model, x[1:N+1,1:dyn.state_dim], base_name="x")
    model[:u] = @variable(model, u[1:N,1:dyn.ctrl_dim], base_name="u")
    model[:ϵ] = @variable(model, ϵ[1:N+1], base_name="ϵ")
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])
    ps = matrix_to_vector_of_vectors(get_position(dyn, model[:x]))

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))) + sum(hps.collision_markup^t * hps.collision_slack * ϵ[t]^2 for t in 1:N))

    # slack variable positivity constraint
    model[:con_ϵ] = @constraint(model, ϵ .>= 0)

    # initial state constraint
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    # dynamic and collision avoidance constraints
    for t in 1:N
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, opt_params.As[t]*xs[t] + opt_params.Bs[t]*us[t] + opt_params.Cs[t] == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
        model[Symbol("collision_avoidance_constraint_$(t)")] = @constraint(model, dot(opt_params.Gs[t], ps[t]) + opt_params.Hs[t] .>= -ϵ[t], base_name="collision_avoidance_constraint_$(t)")
    end
    t = N+1
    model[Symbol("collision_avoidance_constraint_$(t)")] = @constraint(model, dot(opt_params.Gs[t], ps[t]) + opt_params.Hs[t] .>= -ϵ[t], base_name="collision_avoidance_constraint_$(t)")

    # control and velocity constraints
    for t in 1:N
        model[Symbol("control_constraints_upper_$(t)")] = @constraint(model, us[t] <= dyn.control_max, base_name="control_constraints_upper_$(t)")
        model[Symbol("control_constraints_lower_$(t)")] = @constraint(model, dyn.control_min <= us[t] , base_name="control_constraints_lower_$(t)")
        model[Symbol("speed_constraints_upper_$(t)")] = @constraint(model, get_speed(dyn, xs[t], us[t]) .<= dyn.velocity_max , base_name="speed_constraints_upper_$(t)")
        model[Symbol("speed_constraints_lower_$(t)")] = @constraint(model, get_speed(dyn, xs[t], us[t]) .>= dyn.velocity_min , base_name="speed_constraints_lower_$(t)")
    end

    # inconvenience budget constraint
    model[:inconvenience_budget] = @constraint(model, compute_convenience_value(dyn, xs, us, opt_params.goal_state, hps.inconvenience_weights) <= opt_params.inconvenience_budget, base_name="inconvenience_budget")
    InconvenienceProblem(model, xs, us, ϵ, hps, opt_params)
end

function InconvenienceProblem(dyn::IntegratorDynamics, hps::PlannerHyperparameters, opt_params::PlannerOptimizerParams)
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
    model[:x] = @variable(model, x[1:N+1,1:dyn.state_dim], base_name="x")
    model[:u] = @variable(model, u[1:N,1:dyn.ctrl_dim], base_name="u")
    model[:ϵ] = @variable(model, ϵ[1:N+1], base_name="ϵ")
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])
    ps = matrix_to_vector_of_vectors(get_position(dyn, model[:x]))

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))) + sum(hps.collision_markup^t * hps.collision_slack * ϵ[t]^2 for t in 1:N))

    # slack variable positivity constraint
    model[:con_ϵ] = @constraint(model, ϵ .>= 0)

    # initial state constraint
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    # dynamic and collision avoidance constraints
    for t in 1:N
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, opt_params.As[t]*xs[t] + opt_params.Bs[t]*us[t] + opt_params.Cs[t] == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
        model[Symbol("collision_avoidance_constraint_$(t)")] = @constraint(model, dot(opt_params.Gs[t], ps[t]) + opt_params.Hs[t] .>= -ϵ[t], base_name="collision_avoidance_constraint_$(t)")
    end
    t = N+1
    model[Symbol("collision_avoidance_constraint_$(t)")] = @constraint(model, dot(opt_params.Gs[t], ps[t]) + opt_params.Hs[t] .>= -ϵ[t], base_name="collision_avoidance_constraint_$(t)")

    # control and velocity constraints
    for t in 1:N
        model[Symbol("control_constraints_upper_$(t)")] = @constraint(model, us[t] <= dyn.control_max, base_name="control_constraints_upper_$(t)")
        model[Symbol("control_constraints_lower_$(t)")] = @constraint(model, dyn.control_min <= us[t] , base_name="control_constraints_lower_$(t)")
        model[Symbol("speed_constraints_upper_$(t)")] = @constraint(model, get_speed_squared(dyn, xs[t], us[t]) .<= dyn.velocity_max^2 , base_name="speed_constraints_upper_$(t)")
    end

    # inconvenience budget constraint
    model[:inconvenience_budget] = @constraint(model, compute_convenience_value(dyn, xs, us, opt_params.goal_state, hps.inconvenience_weights) <= opt_params.inconvenience_budget, base_name="inconvenience_budget")
    InconvenienceProblem(model, xs, us, ϵ, hps, opt_params)
end

# setup ideal problem
function IdealProblem(dyn::Union{UnicycleDynamics, SingleIntegratorPolar2D}, hps::PlannerHyperparameters, opt_params::PlannerOptimizerParams)
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
    model[:x] = @variable(model, x[1:N+1,1:dyn.state_dim], base_name="x")
    model[:u] = @variable(model, u[1:N,1:dyn.ctrl_dim], base_name="u")
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))))

    # initial state constraint
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    # dynamic constraints
    for t in 1:N
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, opt_params.As[t]*xs[t] + opt_params.Bs[t]*us[t] + opt_params.Cs[t] == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
    end

    # control and speed constraints
    for t in 1:N
        model[Symbol("control_constraints_upper_$(t)")] = @constraint(model, us[t] <= dyn.control_max, base_name="control_constraints_upper_$(t)")
        model[Symbol("control_constraints_lower_$(t)")] = @constraint(model, dyn.control_min <= us[t] , base_name="control_constraints_lower_$(t)")
        model[Symbol("speed_constraints_upper_$(t)")] = @constraint(model, get_speed(dyn, xs[t], us[t]) .<= dyn.velocity_max , base_name="speed_constraints_upper_$(t)")
        model[Symbol("speed_constraints_lower_$(t)")] = @constraint(model, get_speed(dyn, xs[t], us[t]) .>= dyn.velocity_min , base_name="speed_constraints_lower_$(t)")
    end

    IdealProblem(model, xs, us, hps, opt_params)
end

function IdealProblem(dyn::IntegratorDynamics, hps::PlannerHyperparameters, opt_params::PlannerOptimizerParams)
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
    
    model[:x] = @variable(model, x[1:N+1,1:dyn.state_dim], base_name="x")
    model[:u] = @variable(model, u[1:N,1:dyn.ctrl_dim], base_name="u")
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))))

    # initial state constraint
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    # dynamic constraints
    for t in 1:N
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, opt_params.As[t]*xs[t] + opt_params.Bs[t]*us[t] + opt_params.Cs[t] == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
    end

    # control and speed constraints
    for t in 1:N
        model[Symbol("control_constraints_upper_$(t)")] = @constraint(model, us[t] <= dyn.control_max, base_name="control_constraints_upper_$(t)")
        model[Symbol("control_constraints_lower_$(t)")] = @constraint(model, dyn.control_min <= us[t] , base_name="control_constraints_lower_$(t)")
        model[Symbol("speed_constraints_upper_$(t)")] = @constraint(model, get_speed_squared(dyn, xs[t], us[t]) .<= dyn.velocity_max^2 , base_name="speed_constraints_upper_$(t)")
        # model[Symbol("speed_constraints_lower_$(t)")] = @constraint(model, get_speed_squared(dyn, xs[t], us[t]) .>= dyn.velocity_min , base_name="speed_constraints_lower_$(t)")
    end

    IdealProblem(model, xs, us, hps, opt_params)
end

# relinearize dynamics with new states and controls
function update_dynamics_linearization!(opt_params::PlannerOptimizerParams, dyn::Dynamics, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where{T}
    # states = opt_params.previous_states
    # controls = opt_params.previous_controls
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

# relinearize dynamics with new states and controls
function update_dynamics_linearization!(problem::Problem)
    opt_params = problem.opt_params
    dyn = problem.hps.dynamics
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

function update_dynamics_linearization!(problem::Problem, new_states::Vector{Vector{T}}, new_controls::Vector{Vector{T}}) where{T}
    opt_params = problem.opt_params
    opt_params.previous_states = new_states
    opt_params.previous_controls = new_controls
    dyn = problem.hps.dynamics
    N = size(opt_params.As)[1]
    ABCs = linearized_dynamics(dyn, new_states[1:N], new_controls[1:N])
    for (t, (A, B, C)) in enumerate(ABCs)
        opt_params.As[t] = A
        opt_params.Bs[t] = B
        opt_params.Cs[t] = C
    end
end

function update_collision_constraint_linearization!(problem::InconvenienceProblem)
    opt_params = problem.opt_params
    dyn = problem.hps.dynamics
    previous_states = opt_params.previous_states
    other_positions = opt_params.other_positions
    ps = get_position(dyn, previous_states)
    Gs = linearize_collision_avoidance(ps, other_positions)
    Hs = collision_avoidance_constraint(problem.hps.collision_radius, ps, other_positions) - dot.(Gs, ps)
    for (t, (G, H)) in enumerate(zip(Gs, Hs))
        opt_params.Gs[t] = G
        opt_params.Hs[t] = H
    end
end


function update_problem!(problem::IdealProblem)
    model = problem.model
    hps = problem.hps
    opt_params = problem.opt_params
    n = hps.dynamics.state_dim
    m = hps.dynamics.ctrl_dim
    N = hps.time_horizon
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])

    delete_and_unregister(model, :initial_state)
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))))

    for (t, (A,B,C)) in enumerate(zip(opt_params.As, opt_params.Bs, opt_params.Cs))
        delete_and_unregister(model, Symbol("linear_dynamics_constraint_$(t)"))
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, A*xs[t] + B*us[t] + C == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
    end
end

function update_problem!(problem::InconvenienceProblem)
    model = problem.model
    hps = problem.hps
    opt_params = problem.opt_params
    n = hps.dynamics.state_dim
    m = hps.dynamics.ctrl_dim
    N = hps.time_horizon
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])
    ps = matrix_to_vector_of_vectors(get_position(hps.dynamics, model[:x]))

    delete_and_unregister(model, :initial_state)
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))) + sum(hps.collision_markup^t * hps.collision_slack * model[:ϵ][t]^2 for t in 1:N+1))

    # update dynamics constraints
    for (t, (A,B,C)) in enumerate(zip(opt_params.As, opt_params.Bs, opt_params.Cs))
        delete_and_unregister(model, Symbol("linear_dynamics_constraint_$(t)"))
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, A*xs[t] + B*us[t] + C == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
    end

    # update collision avoidance constraints
    for (t, (G, H)) in enumerate(zip(opt_params.Gs, opt_params.Hs))
        delete_and_unregister(model, Symbol("collision_avoidance_constraint_$(t)"))
        model[Symbol("collision_avoidance_constraint_$(t)")] = @constraint(model, dot(opt_params.Gs[t], ps[t]) + opt_params.Hs[t]  .>= -model[:ϵ][t], base_name="collision_avoidance_constraint_$(t)")
    end

    # update inconvenience budget constraint
    # set_normalized_rhs(model[:inconvenience_budget], opt_params.inconvenience_budget)     # this had issues
    delete_and_unregister(model, :inconvenience_budget)
    model[:inconvenience_budget] = @constraint(model, compute_convenience_value(problem.hps.dynamics, xs, us, opt_params.goal_state, hps.inconvenience_weights) <= opt_params.inconvenience_budget, base_name="inconvenience_budget")
end

function update_problem!(problem::InconvenienceProblem, constant_velo_agents::Vector{ConstantVeloAgent})
    model = problem.model
    hps = problem.hps
    opt_params = problem.opt_params
    n = hps.dynamics.state_dim
    m = hps.dynamics.ctrl_dim
    N = hps.time_horizon
    xs = matrix_to_vector_of_vectors(model[:x])
    us = matrix_to_vector_of_vectors(model[:u])
    ps = matrix_to_vector_of_vectors(get_position(hps.dynamics, model[:x]))
    ego_ps = get_position(hps.dynamics, opt_params.previous_states)
    N_velo_agents = length(constant_velo_agents)

    constant_velo_pos = Vector{Vector}(undef, N_velo_agents)
    constant_velo_Gs = Vector{Vector}(undef, N_velo_agents)
    constant_velo_Hs = Vector{Vector}(undef, N_velo_agents)

    for i in 1:N_velo_agents
        constant_velo_pos[i] = get_constant_velocity_agent_positions(problem, constant_velo_agents[i])
        constant_velo_Gs[i] = linearize_collision_avoidance(ego_ps, constant_velo_pos[i])
        constant_velo_Hs[i] = collision_avoidance_constraint(problem.hps.collision_radius, ego_ps, constant_velo_pos[i]) - dot.(constant_velo_Gs[i], ego_ps)
    end

    delete_and_unregister(model, :initial_state)
    model[:initial_state] = @constraint(model, xs[1] == opt_params.initial_state, base_name="initial_state")

    @objective(model, Min, compute_running_quadratic_cost(xs[1:N], hps.Q, markup=hps.markup) + compute_running_quadratic_cost(us[1:N], hps.R, markup=hps.markup) + compute_quadratic_error_cost(xs[end], opt_params.goal_state, hps.Qt) + hps.trust_region_weight * (compute_running_quadratic_cost(xs - opt_params.previous_states, Matrix{Float64}(I, n, n)) + compute_running_quadratic_cost(us - opt_params.previous_controls, Matrix{Float64}(I, m, m))) + sum(hps.collision_markup^t * hps.collision_slack * model[:ϵ][t]^2 for t in 1:N+1))

    # update dynamics constraints
    for (t, (A,B,C)) in enumerate(zip(opt_params.As, opt_params.Bs, opt_params.Cs))
        delete_and_unregister(model, Symbol("linear_dynamics_constraint_$(t)"))
        model[Symbol("linear_dynamics_constraint_$(t)")] = @constraint(model, A*xs[t] + B*us[t] + C == xs[t+1], base_name="linear_dynamics_constraint_$(t)")
    end

    # update collision avoidance constraints
    for (t, (G, H)) in enumerate(zip(opt_params.Gs, opt_params.Hs))
        delete_and_unregister(model, Symbol("collision_avoidance_constraint_$(t)"))
        model[Symbol("collision_avoidance_constraint_$(t)")] = @constraint(model, dot(opt_params.Gs[t], ps[t]) + opt_params.Hs[t]  .>= -model[:ϵ][t], base_name="collision_avoidance_constraint_$(t)")
        for i in 1:N_velo_agents
            delete_and_unregister(model, Symbol("constant_velo_avoidance_agent_$(i)_$(t)"))
            model[Symbol("constant_velo_avoidance_agent_$(i)_$(t)")] = @constraint(model, dot(constant_velo_Gs[i][t], ego_ps[t]) + constant_velo_Hs[i][t] .>= -model[:ϵ][t], base_name="constant_velo_avoidance_agent_$(i)_$(t)")
        end
    end

    # update inconvenience budget constraint
    # set_normalized_rhs(model[:inconvenience_budget], opt_params.inconvenience_budget)     # this had issues
    delete_and_unregister(model, :inconvenience_budget)
    model[:inconvenience_budget] = @constraint(model, compute_convenience_value(problem.hps.dynamics, xs, us, opt_params.goal_state, hps.inconvenience_weights) <= opt_params.inconvenience_budget, base_name="inconvenience_budget")
end

function solve(problem::IdealProblem; iterations=5, verbose=false, keep_history=false)
    MOI.set(problem.model, MOI.Silent(), !verbose)
    if keep_history
        xs = []
        us = []
    end
    for i in 1:iterations
        update_dynamics_linearization!(problem)
        update_problem!(problem)
        MOI.set(problem.model, MOI.Silent(), !verbose)
        optimize!(problem.model);
        # TODO check for solution feasibility
        problem.opt_params.previous_states = matrix_to_vector_of_vectors(value.(problem.model[:x]))
        problem.opt_params.previous_controls = matrix_to_vector_of_vectors(value.(problem.model[:u]))
        if keep_history
            push!(xs, value.(problem.model[:x]))
            push!(us, value.(problem.model[:u]))
        end
    end
    if keep_history
        return problem, xs, us
    end

    # update inconvenience budget
    update_convenience_budget!(problem)

    return problem, [value.(problem.model[:x])], [value.(problem.model[:u])]
end

function solve(problem::InconvenienceProblem; iterations=5, verbose=false, keep_history=false)
    MOI.set(problem.model, MOI.Silent(), !verbose)
    if keep_history
        xs = []
        us = []
    end
    for i in 1:iterations
        update_dynamics_linearization!(problem)
        update_collision_constraint_linearization!(problem)
        update_problem!(problem)
        MOI.set(problem.model, MOI.Silent(), !verbose)
        optimize!(problem.model);
        problem.opt_params.previous_states = matrix_to_vector_of_vectors(value.(problem.model[:x]))
        problem.opt_params.previous_controls = matrix_to_vector_of_vectors(value.(problem.model[:u]))
        if keep_history
            push!(xs, value.(problem.model[:x]))
            push!(us, value.(problem.model[:u]))
        end
    end
    if keep_history
        return problem, xs, us
    end
    return problem, [value.(problem.model[:x])], [value.(problem.model[:u])]
end

function solve(problem::InconvenienceProblem, constant_velo_agents::Vector{ConstantVeloAgent}; iterations=5, verbose=false, keep_history=false)
    MOI.set(problem.model, MOI.Silent(), !verbose)
    if keep_history
        xs = []
        us = []
    end
    for i in 1:iterations
        update_dynamics_linearization!(problem)
        update_collision_constraint_linearization!(problem)
        update_problem!(problem, constant_velo_agents)
        MOI.set(problem.model, MOI.Silent(), !verbose)
        optimize!(problem.model);
        problem.opt_params.previous_states = matrix_to_vector_of_vectors(value.(problem.model[:x]))
        problem.opt_params.previous_controls = matrix_to_vector_of_vectors(value.(problem.model[:u]))
        if keep_history
            push!(xs, value.(problem.model[:x]))
            push!(us, value.(problem.model[:u]))
        end
    end
    if keep_history
        return problem, xs, us
    end
end

mutable struct AgentPlanner <: Planner
    ideal::IdealProblem
    incon::InconvenienceProblem
end

@with_kw mutable struct InteractionPlanner
    ego_planner::AgentPlanner
    other_planner::AgentPlanner
end

function InteractionPlanner(ego_hps::PlannerHyperparameters,
                            other_hps::PlannerHyperparameters,
                            ego_initial_state::Vector{T},
                            other_initial_state::Vector{T},
                            ego_goal_state::Vector{T},
                            other_goal_state::Vector{T},
                            solver::String) where {T}
    ego = ego_hps.dynamics
    other = other_hps.dynamics

    # setting up ego ideal planner
    ego_opt_params = PlannerOptimizerParams(ego,
                        ego_hps,
                        ego_initial_state,
                        ego_goal_state,
                        solver
                        )
    ego_opt_params.initial_state = ego_initial_state
    ego_opt_params.goal_state = ego_goal_state
    ego_ideal_problem = IdealProblem(ego, ego_hps, ego_opt_params)

    # setting up other ideal planner
    other_opt_params = PlannerOptimizerParams(other,
                        other_hps,
                        other_initial_state,
                        other_goal_state,
                        solver
                        )
    other_opt_params.initial_state = other_initial_state
    other_opt_params.goal_state = other_goal_state
    other_ideal_problem = IdealProblem(other, other_hps, other_opt_params)

    # solve ego and other ideal problem
    _, ego_ideal_xs, ego_ideal_us = solve(ego_ideal_problem, iterations=3, verbose=false, keep_history=false)
    _, other_ideal_xs, other_ideal_us = solve(other_ideal_problem, iterations=3, verbose=false, keep_history=false)

    # update previous states and controls with ideal solution
    # ego_opt_params.previous_states = matrix_to_vector_of_vectors(ego_ideal_xs[end])
    # ego_opt_params.previous_controls = matrix_to_vector_of_vectors(ego_ideal_us[end])
    # other_opt_params.previous_states = matrix_to_vector_of_vectors(other_ideal_xs[end])
    # other_opt_params.previous_controls = matrix_to_vector_of_vectors(other_ideal_us[end])
    ego_ps = get_position(ego, ego_opt_params.previous_states)
    other_ps = get_position(other, other_opt_params.previous_states)

    # set up ego inconvenience planner
    Gs = linearize_collision_avoidance(ego_ps, other_ps)
    Hs = collision_avoidance_constraint(ego_hps.collision_radius, ego_ps, other_ps) - dot.(Gs, ego_ps)
    ego_opt_params.Gs = Gs
    ego_opt_params.Hs = Hs
    ego_opt_params.other_positions = other_ps
    ego_incon_problem = InconvenienceProblem(ego, ego_hps, ego_opt_params)

    # set up other inconvenience planner
    # update previous states and controls with ideal solution
    Gs = linearize_collision_avoidance(other_ps, ego_ps)
    Hs = collision_avoidance_constraint(other_hps.collision_radius, other_ps, ego_ps) - dot.(Gs, other_ps)
    other_opt_params.Gs = Gs
    other_opt_params.Hs = Hs
    other_opt_params.other_positions = ego_ps
    other_incon_problem = InconvenienceProblem(other, other_hps, other_opt_params)

    ego_planner = AgentPlanner(ego_ideal_problem, ego_incon_problem)
    other_planner = AgentPlanner(other_ideal_problem, other_incon_problem)

    InteractionPlanner(ego_planner, other_planner)
end

function update_agent!(agent::AgentPlanner, other::AgentPlanner)
    agent.incon.opt_params.other_positions = get_position(other.incon.hps.dynamics, other.incon.opt_params.previous_states)
    update_collision_constraint_linearization!(agent.incon)
    update_dynamics_linearization!(agent.incon)
    update_problem!(agent.incon)
    agent.incon
end

function ibr(ip::InteractionPlanner, iterations::Int64, leader="ego"::String)
    if leader != "ego"                       # determine which agent solves leader
        leader_agent = ip.other_planner
        follower_agent = ip.ego_planner
    else
        leader_agent = ip.ego_planner
        follower_agent = ip.other_planner
    end

    # ideal_path computation for ego and other
    # opt_params.inconvenience_budget for ego and other

    for i in 1:iterations
        
        # linearize collision avoidance constraints
        # linearize dynamics
        # update JuMP model
        # update previous state and controls with latest solution
        leader_agent.incon.opt_params.other_positions = get_position(follower_agent.incon.hps.dynamics, follower_agent.incon.opt_params.previous_states)
        solve(leader_agent.incon, iterations=1)
        follower_agent.incon.opt_params.other_positions = get_position(leader_agent.incon.hps.dynamics, leader_agent.incon.opt_params.previous_states)
        solve(follower_agent.incon, iterations=1)
    end

    # ip, value.(ip.ego_planner.incon.xs), value.(ip.ego_planner.incon.us), value.(ip.ego_planner.incon.us)[1]
    ip
end

function ibr(ip::InteractionPlanner, iterations::Int64, leader="ego"::String, constant_velo_agents::ConstantVeloAgent...)
    if leader != "ego"                       # determine which agent solves leader
        leader_agent = ip.other_planner
        follower_agent = ip.ego_planner
    else
        leader_agent = ip.ego_planner
        follower_agent = ip.other_planner
    end

    velo_agents = collect(constant_velo_agents)

    # ideal_path computation for ego and other
    # opt_params.inconvenience_budget for ego and other

    for i in 1:iterations
        
        # linearize collision avoidance constraints
        # linearize dynamics
        # update JuMP model
        # update previous state and controls with latest solution
        leader_agent.incon.opt_params.other_positions = get_position(follower_agent.incon.hps.dynamics, follower_agent.incon.opt_params.previous_states)
        solve(leader_agent.incon, velo_agents, iterations=1)
        follower_agent.incon.opt_params.other_positions = get_position(leader_agent.incon.hps.dynamics, leader_agent.incon.opt_params.previous_states)
        solve(follower_agent.incon, velo_agents, iterations=1)
    end

    # ip, value.(ip.ego_planner.incon.xs), value.(ip.ego_planner.incon.us), value.(ip.ego_planner.incon.us)[1]
    ip
end

@with_kw mutable struct SaveData
    previous_ips::Vector{InteractionPlanner}
end

function ibr_save(ip::InteractionPlanner, iterations::Int64, leader="ego"::String)     # for param sweep saving
    if leader != "ego"                       # determine which agent solves leader
        leader_agent = ip.other_planner
        follower_agent = ip.ego_planner
    else
        leader_agent = ip.ego_planner
        follower_agent = ip.other_planner
    end

    data = SaveData(Vector{InteractionPlanner}(undef, iterations))

    # ideal_path computation for ego and other
    # opt_params.inconvenience_budget for ego and other

    leader_xs = Vector{Matrix{Float64}}(undef, iterations)
    follower_xs = Vector{Matrix{Float64}}(undef, iterations)
    leader_us = Vector{Matrix{Float64}}(undef, iterations)
    follower_us = Vector{Matrix{Float64}}(undef, iterations)


    for i in 1:iterations
        
        # linearize collision avoidance constraints
        # linearize dynamics
        # update JuMP model
        # update previous state and controls with latest solution
        leader_agent.incon.opt_params.other_positions = get_position(follower_agent.incon.hps.dynamics, follower_agent.incon.opt_params.previous_states)
        _, xs_, us_ = solve(leader_agent.incon, iterations=1)
        leader_xs[i] = xs_[1]
        leader_us[i] = us_[1]

        follower_agent.incon.opt_params.other_positions = get_position(leader_agent.incon.hps.dynamics, leader_agent.incon.opt_params.previous_states)
        _, xs_, us_ = solve(follower_agent.incon, iterations=1)
        follower_xs[i] = xs_[1]
        follower_us[i] = us_[1]
        data.previous_ips[i] = deepcopy(ip)     # store data at each iteration

    end

    # ip, value.(ip.ego_planner.incon.xs), value.(ip.ego_planner.incon.us), value.(ip.ego_planner.incon.us)[1]
    data, leader_xs, leader_us, follower_xs, follower_us
end

function ibr_mpc(ip::InteractionPlanner, iterations::Int64, leader="ego"::String)
    # ideal_path computation for ego and other
    # update previous states and controls with ego's  ideal solution using initial straightline trajectory
    ego_dyn = ip.ego_planner.ideal.hps.dynamics
    ego_initial_state = ip.ego_planner.ideal.opt_params.initial_state
    ego_goal_state = ip.ego_planner.ideal.opt_params.goal_state
    ego_initial_speed = get_speed(ego_dyn, ego_initial_state, ip.ego_planner.ideal.opt_params.previous_controls[1])[1]
    ip.ego_planner.ideal.opt_params.previous_states = matrix_to_vector_of_vectors(initial_straight_trajectory(ego_dyn, ego_initial_state, ego_goal_state, ego_initial_speed, ip.ego_planner.ideal.hps.time_horizon)[1])     # populates ego states and controls w/ straight line trajectory
    # _, _= initial_straight_trajectory(ego_dyn, ego_initial_state, ego_goal_state, ego_initial_speed, ip.ego_planner.ideal.hps.time_horizon)     # populates ego states and controls w/
    ip.ego_planner.ideal.opt_params.previous_controls = matrix_to_vector_of_vectors(initial_straight_trajectory(ego_dyn, ego_initial_state, ego_goal_state, ego_initial_speed, ip.ego_planner.ideal.hps.time_horizon)[2])


    # update previous_states/controls for other agent w/ straight line trajectory from new initial state.
    other_dyn = ip.other_planner.ideal.hps.dynamics
    other_initial_state = ip.other_planner.ideal.opt_params.initial_state
    other_goal_state = ip.other_planner.ideal.opt_params.goal_state
    other_initial_speed = get_speed(other_dyn, other_initial_state, ip.other_planner.ideal.opt_params.previous_controls[1])[1]
    ip.other_planner.ideal.opt_params.previous_states = matrix_to_vector_of_vectors(initial_straight_trajectory(other_dyn, other_initial_state, other_goal_state, other_initial_speed, ip.other_planner.ideal.hps.time_horizon)[1])     # populates other states and controls w/ straight line trajectory
    # _, _= initial_straight_trajectory(other_dyn, other_initial_state, other_goal_state, other_initial_speed, ip.other_planner.ideal.hps.time_horizon)     # populates other states and controls w/
    ip.other_planner.ideal.opt_params.previous_controls = matrix_to_vector_of_vectors(initial_straight_trajectory(other_dyn, other_initial_state, other_goal_state, other_initial_speed, ip.other_planner.ideal.hps.time_horizon)[2])



    if leader != "ego"                       # determine which agent solves leader
        leader_agent = ip.other_planner
        follower_agent = ip.ego_planner
    else
        leader_agent = ip.ego_planner
        follower_agent = ip.other_planner
    end

    solve(leader_agent.ideal, iterations=3)
    solve(follower_agent.ideal, iterations=3)

    for i in 1:iterations
        
        # linearize collision avoidance constraints
        # linearize dynamics
        # update JuMP model
        # update previous state and controls with latest solution
        leader_agent.incon.opt_params.other_positions = get_position(follower_agent.incon.hps.dynamics, follower_agent.incon.opt_params.previous_states)
        if iterations == 1  # creates a non ibr optimal control problem
            solve(leader_agent.incon, iterations=5)
        else
            solve(leader_agent.incon, iterations=1)
        end

        follower_agent.incon.opt_params.other_positions = get_position(leader_agent.incon.hps.dynamics, leader_agent.incon.opt_params.previous_states)
        if iterations == 1  # creates a non ibr optimal control problem
            solve(follower_agent.incon, iterations=5)
        else
            solve(follower_agent.incon, iterations=1)
        end
    end

    # ip, value.(ip.ego_planner.incon.xs), value.(ip.ego_planner.incon.us), value.(ip.ego_planner.incon.us)[1]
    ip, value.(ip.ego_planner.incon.model[:x]), value.(ip.ego_planner.incon.model[:u])
end

function ibr_mpc(ip::InteractionPlanner, constant_velo_agents::Vector{ConstantVeloAgent}, iterations::Int64, leader="ego"::String)
    # ideal_path computation for ego and other
    # update previous states and controls with ego's  ideal solution using initial straightline trajectory
    ego_dyn = ip.ego_planner.ideal.hps.dynamics
    ego_initial_state = ip.ego_planner.ideal.opt_params.initial_state
    ego_goal_state = ip.ego_planner.ideal.opt_params.goal_state
    ego_initial_speed = get_speed(ego_dyn, ego_initial_state, ip.ego_planner.ideal.opt_params.previous_controls[1])[1]
    ip.ego_planner.ideal.opt_params.previous_states = matrix_to_vector_of_vectors(initial_straight_trajectory(ego_dyn, ego_initial_state, ego_goal_state, ego_initial_speed, ip.ego_planner.ideal.hps.time_horizon)[1])     # populates ego states and controls w/ straight line trajectory
    # _, _= initial_straight_trajectory(ego_dyn, ego_initial_state, ego_goal_state, ego_initial_speed, ip.ego_planner.ideal.hps.time_horizon)     # populates ego states and controls w/
    ip.ego_planner.ideal.opt_params.previous_controls = matrix_to_vector_of_vectors(initial_straight_trajectory(ego_dyn, ego_initial_state, ego_goal_state, ego_initial_speed, ip.ego_planner.ideal.hps.time_horizon)[2])


    # update previous_states/controls for other agent w/ straight line trajectory from new initial state.
    other_dyn = ip.other_planner.ideal.hps.dynamics
    other_initial_state = ip.other_planner.ideal.opt_params.initial_state
    other_goal_state = ip.other_planner.ideal.opt_params.goal_state
    other_initial_speed = get_speed(other_dyn, other_initial_state, ip.other_planner.ideal.opt_params.previous_controls[1])[1]
    ip.other_planner.ideal.opt_params.previous_states = matrix_to_vector_of_vectors(initial_straight_trajectory(other_dyn, other_initial_state, other_goal_state, other_initial_speed, ip.other_planner.ideal.hps.time_horizon)[1])     # populates other states and controls w/ straight line trajectory
    # _, _= initial_straight_trajectory(other_dyn, other_initial_state, other_goal_state, other_initial_speed, ip.other_planner.ideal.hps.time_horizon)     # populates other states and controls w/
    ip.other_planner.ideal.opt_params.previous_controls = matrix_to_vector_of_vectors(initial_straight_trajectory(other_dyn, other_initial_state, other_goal_state, other_initial_speed, ip.other_planner.ideal.hps.time_horizon)[2])



    if leader != "ego"                       # determine which agent solves leader
        leader_agent = ip.other_planner
        follower_agent = ip.ego_planner
    else
        leader_agent = ip.ego_planner
        follower_agent = ip.other_planner
    end

    solve(leader_agent.ideal, iterations=3)
    solve(follower_agent.ideal, iterations=3)

    for i in 1:iterations
        
        # linearize collision avoidance constraints
        # linearize dynamics
        # update JuMP model
        # update previous state and controls with latest solution
        leader_agent.incon.opt_params.other_positions = get_position(follower_agent.incon.hps.dynamics, follower_agent.incon.opt_params.previous_states)
        solve(leader_agent.incon, constant_velo_agents, iterations=1)
        follower_agent.incon.opt_params.other_positions = get_position(leader_agent.incon.hps.dynamics, leader_agent.incon.opt_params.previous_states)
        solve(follower_agent.incon, constant_velo_agents, iterations=1)
    end

    # ip, value.(ip.ego_planner.incon.xs), value.(ip.ego_planner.incon.us), value.(ip.ego_planner.incon.us)[1]
    ip, value.(ip.ego_planner.incon.model[:x]), value.(ip.ego_planner.incon.model[:u])
end