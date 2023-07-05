using LinearAlgebra
using EllipsisNotation
using ForwardDiff

abstract type Dynamics end

struct SingleIntegrator2D{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64
    ctrl_dim::Int64
    control_lim::VecOrMat{T}
end

function step(dyn::SingleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T}
    A = [1. 0. ; 0. 1.]
    B = [dyn.dt 0.; 0 dyn.dt]
    A * state + B * control 
end

get_position(dyn::SingleIntegrator2D, state::VecOrMat{T}) where {T} = state
get_velocity(dyn::SingleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = control

function initial_straight_trajectory(dyn::SingleIntegrator2D, start_position::VecOrMat{T}, end_position::VecOrMat{T}, initial_speed::T, T_horizon::Int64) where {T}
    x0, y0 = start_position
    xg, yg = end_position
    initial_heading = atan(yg - y0, xg - x0)
    vx, vy = initial_speed * cos(initial_heading), initial_speed * sin(initial_heading)
    initial_state = [x0; y0]
    states = Array{Float64}(undef, T_horizon+1, dyn.state_dim)
    states[1,:] = initial_state
    controls = ones(Float64, T_horizon, dyn.ctrl_dim) * diagm([vx, vy])
    for t = 1:T_horizon
        states[t+1,:] = step(dyn, states[t,:], controls[t,:])
    end
    states, controls
end

function linearized_dynamics(dyn::SingleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T}
    A = [1. 0. ; 0. 1.]
    B = [dyn.dt 0.; 0 dyn.dt]
    C = zeros(Float64, dyn.state_dim)
    A, B, C
end


struct DoubleIntegrator2D{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64
    ctrl_dim::Int64
    control_lim::VecOrMat{T}
end

function step(dyn::DoubleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T}
    A = [1. 0. dyn.dt 0.; 0. 1. 0. dyn.dt; 0. 0. 1. 0.; 0. 0. 0. 1.]
    B = [0.5 * dyn.dt^2 0.; 0. 0.5 * dyn.dt^2; dyn.dt 0.; 0. dyn.dt]
    A * state + B * control 
end
get_position(dyn::DoubleIntegrator2D, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::DoubleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = state[..,end-1:end]

function initial_straight_trajectory(dyn::DoubleIntegrator2D, start_position::VecOrMat{T}, end_position::VecOrMat{T}, initial_speed::T, T_horizon::Int64) where {T}
    x0, y0 = start_position
    xg, yg = end_position
    initial_heading = atan(yg - y0, xg - x0)
    vx, vy = initial_speed * cos(initial_heading), initial_speed * sin(initial_heading)
    initial_state = [x0; y0; vx; vy]
    states = Array{Float64}(undef, T_horizon+1, dyn.state_dim)
    states[1,:] = initial_state
    controls = zeros(Float64, T_horizon, dyn.ctrl_dim)
    for t = 1:T_horizon
        states[t+1,:] = step(dyn, states[t,:], [0.;0.])
    end
    states, controls
end

function linearized_dynamics(dyn::DoubleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T}
    A = [1. 0. dyn.dt 0.; 0. 1. 0. dyn.dt; 0. 0. 1. 0.; 0. 0. 0. 1.]
    B = [0.5 * dyn.dt^2 0.; 0. 0.5 * dyn.dt^2; dyn.dt 0.; 0. dyn.dt]
    C = zeros(Float64, dyn.state_dim)
    A, B, C
end

struct UnicycleDynamics{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64
    ctrl_dim::Int64
    control_lim::Vector{T}
end

function step(dyn::UnicycleDynamics, state, control)
    dt = 0.1
    x, y, theta, = state
    w, v = control
    eps = 1E-2
    w_ = abs(w) < eps ? 1.0 :  w
    dx = abs(w) < eps ? v * dt * cos(theta) : v / w_ * (sin(theta + w * dt) - sin(theta))
    dy = abs(w) < eps ? v * dt * sin(theta) : -v / w_ * (cos(theta + w * dt) - cos(theta))
    state + [dx; dy; w*dt]
end

get_position(dyn::UnicycleDynamics, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::UnicycleDynamics, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = control[..,end:end]

function initial_straight_trajectory(dyn::UnicycleDynamics, start_position::VecOrMat{T}, end_position::VecOrMat{T}, initial_speed::T, T_horizon::Int64) where {T}
    x0, y0 = start_position
    xg, yg = end_position
    initial_heading = atan(yg - y0, xg - x0)
    initial_state = [x0; y0; initial_heading]
    states = Array{Float64}(undef, T_horizon+1, dyn.state_dim)
    states[1,:] = initial_state
    controls = ones(Float64, T_horizon, dyn.ctrl_dim) * diagm([0., initial_speed])
    for t = 1:T_horizon
        states[t+1,:] = step(dyn, states[t,:], controls[t,:])
    end
    states, controls
end

function linearized_dynamics(dyn::UnicycleDynamics, state::VecOrMat{T}, control::VecOrMat{T}) where {T}
    A = ForwardDiff.jacobian(state -> step(dyn, state, control), state)
    B = ForwardDiff.jacobian(control -> step(dyn, state, control), control)
    C = step(dyn, state, control) - A * state - B * control
    A, B, C
end

struct DynamicallyExtendedUnicycleDynamics{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64
    ctrl_dim::Int64
    control_lim::VecOrMat{T}
end

function step(dyn::DynamicallyExtendedUnicycleDynamics, state, control)
    dt = 0.1
    x, y, theta, v = state
    w, a = control
    eps = 1E-2
    w_ = abs(w) < eps ? 1.0 :  w
    dx = abs(w) < eps ? (v * dt + 0.5 * a * dt^2) * cos(theta) : (w_ * (a * dt + v) * sin(theta + dt * w_) + a * cos(theta + dt * w_) - a * cos(theta) - v * w_ * sin(theta))/w_^2
    dy = abs(w) < eps ? (v * dt + 0.5 * a * dt^2) * sin(theta) : (-w_ * (a * dt + v) * cos(theta + dt * w_) + a * (sin(theta + dt * w_) - sin(theta)) + v * w_ * cos(theta))/w_^2

    state + [dx; dy; w * dt; a * dt]
end

get_position(dyn::DynamicallyExtendedUnicycleDynamics, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::DynamicallyExtendedUnicycleDynamics, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = state[..,end:end]

function initial_straight_trajectory(dyn::DynamicallyExtendedUnicycleDynamics, start_position::VecOrMat{T}, end_position::VecOrMat{T}, initial_speed::T, T_horizon::Int64) where {T}
    x0, y0 = start_position
    xg, yg = end_position
    initial_heading = atan(yg - y0, xg - x0)
    initial_state = [x0; y0; initial_heading; initial_speed]
    states = Array{Float64}(undef, T_horizon+1, dyn.state_dim)
    states[1,:] = initial_state
    controls = zeros(Float64, T_horizon, dyn.ctrl_dim)
    for t = 1:T_horizon
        states[t+1,:] = step(dyn, states[t,:], [0.;0.])
    end
    states, controls
end

function linearized_dynamics(dyn::DynamicallyExtendedUnicycleDynamics, state::VecOrMat{T}, control::VecOrMat{T}) where {T}

    A = ForwardDiff.jacobian(state -> step(dyn, state, control), state)
    B = ForwardDiff.jacobian(control -> step(dyn, state, control), control)
    C = step(dyn, state, control) - A * state - B * control
    A, B, C
end
