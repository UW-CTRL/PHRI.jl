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
