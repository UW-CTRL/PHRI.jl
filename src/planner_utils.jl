using LinearAlgebra
using ForwardDiff

compute_quadratic_error_cost(state::VecOrMat{T}, goal::VecOrMat{T}, Qt::Matrix{T}) where {T} =  transpose(state - goal) * Qt * (state - goal)

compute_quadratic_cost(state::VecOrMat{T}, Q::Matrix{T}) where {T} = compute_quadratic_error_cost(state, zeros(Float64, size(state)), Q)

compute_running_quadratic_cost(states::VecOrMat{T}, Q::Matrix{T}) where {T} = sum([compute_quadratic_cost(states[i,:], Q) for i in 1:size(states)[1]])

function compute_total_difference_squared(time_series::VecOrMat{T}) where {T}
    dx = diff(time_series, dims=1)
    n = size(time_series)[end]
    Q = Matrix{Float64}(I, n, n)
    compute_running_quadratic_cost(dx, Q)
end    

function compute_convenience_value(dynamics::Dynamics, states::VecOrMat{T}, controls::VecOrMat{T}, goal::VecOrMat{T}, inconvenience_weights::VecOrMat{T}) where {T}
    position = get_position(dynamics, states)
    velocity = get_velocity(dynamics, states, controls)
    n = dynamics.state_dim
    total_squared_distance = compute_total_difference_squared(position)
    total_squared_acceleration = compute_total_difference_squared(velocity)
    distance_squared_from_goal = compute_quadratic_error_cost(states[end,:], goal, Matrix{Float64}(I, n, n))
    v = [total_squared_distance, total_squared_acceleration, distance_squared_from_goal]
    dot(inconvenience_weights, v)
end


function collision_avoidance_constraint(radius::T, ego_dyn::Dynamics, ego_state::VecOrMat, other_dyn::Dynamics, other_state::VecOrMat) where {T}
    ego_position = get_position(ego_dyn, ego_state)
    other_position = get_position(other_dyn, other_state)
    ego_position - other_position
    norm(ego_position - other_position, 2)^2 - radius^2
end

function collision_avoidance_constraint(radius::T, ego_position::VecOrMat, other_position::VecOrMat) where {T}
    sum((ego_position - other_position).^2) - radius^2
end

function linearize_collision_avoidance(ego_dyn::Dynamics, ego_state::VecOrMat, other_dyn::Dynamics, other_state::VecOrMat)
    ego_position = get_position(ego_dyn, ego_state)
    other_position = get_position(other_dyn, other_state)
    2 * (ego_position - other_position)
end 

function linearize_collision_avoidance(ego_position::VecOrMat, other_position::VecOrMat)
    2 * (ego_position - other_position)
end 