using LinearAlgebra
using ForwardDiff
using Parameters
using JuMP, HiGHS, ECOS

NumberOrVariable = Union{Number, VariableRef}

function delete_and_unregister(model::Model, tag::Symbol)
    delete(model, model[tag])
    unregister(model, tag)
end

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
    speed = get_speed(dynamics, states[1:end-1], controls)
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

function update_convenience_budget!(problem::Problem)
    dyn = problem.hps.dynamics
    xs = matrix_to_vector_of_vectors(value.(problem.model[:x]))
    us = matrix_to_vector_of_vectors(value.(problem.model[:u]))
    goal = problem.opt_params.goal_state
    weights = problem.hps.inconvenience_weights

    convenience_value = compute_convenience_value(dyn, xs, us, goal, weights)

    problem.opt_params.inconvenience_budget = convenience_value * (1 + problem.hps.inconvenience_ratio)
end

function add_constant_velocity_agent(problem::InconvenienceProblem, constant_velo_agent::ConstantVeloAgent)
    opt_params = problem.opt_params
    dyn = problem.hps.dynamics
    N = problem.hps.time_horizon
    dt = problem.hps.dynamics.dt
    pos = constant_velo_agent.pos
    velo_vector = constant_velo_agent.velo
    model = problem.model

    previous_states = opt_params.previous_states
    ps = get_position(dyn, previous_states)

    other_positions = get_constant_velocity_agent_positions(problem, constant_velo_agent)

    Gs = linearize_collision_avoidance(ps, other_positions)
    Hs = collision_avoidance_constraint(problem.hps.collision_radius, ps, other_positions) - dot.(Gs, ps)

    for t in 1:N+1
    model[Symbol("constant_velo_avoidance_agent_1_$(t)")] = @constraint(model, dot(Gs[t], ps[t]) + Hs[t] .>= -model[:ϵ], base_name="constant_velo_avoidance_agent_1_$(t)")
    end      
end

function get_constant_velocity_agent_positions(problem::Problem, constant_velo_agent::ConstantVeloAgent)
    N = problem.hps.time_horizon
    dt = problem.hps.dynamics.dt
    pos = constant_velo_agent.pos
    velo_vector = constant_velo_agent.velo

    positions = [pos + velo_vector * dt * i for i in 0:N]
end