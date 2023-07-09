using LinearAlgebra
using EllipsisNotation
using ForwardDiff
using Parameters

abstract type Dynamics end
abstract type IntegratorDynamics <: Dynamics end
abstract type UnicycleDynamics <: Dynamics end

get_speed(dyn::Dynamics, state::Vector{T}, control::Vector{T}) where {T} = norm.(eachrow(get_velocity(dyn, state, control)))

@with_kw struct SingleIntegrator2D{T} <: IntegratorDynamics where {T}
    dt::T
    state_dim::Int64 = 2    # [x, y]
    ctrl_dim::Int64 = 2     # [ẋ, ẏ]   
    velocity_min::T = 0.
    velocity_max::T
    control_min::Vector{T}
    control_max::Vector{T}
end

# assuming control_lims are symmetric
SingleIntegrator2D(dt::T, velocity_max::T, control_lim::Vector{T}) where {T} = SingleIntegrator2D(dt=dt, velocity_max=velocity_max, control_min=-control_lim, control_max=control_lim)


function step(dyn::SingleIntegrator2D, state::Vector{T}, control::Vector{T}) where {T}
    A = [1. 0. ; 0. 1.]
    B = [dyn.dt 0.; 0 dyn.dt]
    A * state + B * control 
end

get_position(dyn::SingleIntegrator2D, state::VecOrMat{T}) where {T} = state
get_velocity(dyn::SingleIntegrator2D, state::VecOrMat{T}, control::Vector{T}) where {T} = control

function initial_straight_trajectory(dyn::SingleIntegrator2D, start_position::Vector{T}, end_position::Vector{T}, initial_speed::T, T_horizon::Int64) where {T}
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

function linearized_dynamics(dyn::SingleIntegrator2D, state::Vector{T}, control::Vector{T}) where {T}
    A = [1. 0. ; 0. 1.]
    B = [dyn.dt 0.; 0 dyn.dt]
    C = zeros(Float64, dyn.state_dim)
    A, B, C
end

@with_kw struct DoubleIntegrator2D{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64 = 4    # [x, y, ẋ, ẏ]
    ctrl_dim::Int64 = 2     # [ẍ, ÿ]
    velocity_min::T = 0.
    velocity_max::T
    control_min::Vector{T}
    control_max::Vector{T}
end

# assuming control_lims are symmetric
DoubleIntegrator2D(dt::T, velocity_max::T, control_lim::Vector{T}) where {T} = DoubleIntegrator2D(dt=dt, velocity_max=velocity_max, control_min=-control_lim, control_max=control_lim)

function step(dyn::DoubleIntegrator2D, state::Vector{T}, control::Vector{T}) where {T}
    A = [1. 0. dyn.dt 0.; 0. 1. 0. dyn.dt; 0. 0. 1. 0.; 0. 0. 0. 1.]
    B = [0.5 * dyn.dt^2 0.; 0. 0.5 * dyn.dt^2; dyn.dt 0.; 0. dyn.dt]
    A * state + B * control 
end
get_position(dyn::DoubleIntegrator2D, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::DoubleIntegrator2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = state[..,3:4]

function initial_straight_trajectory(dyn::DoubleIntegrator2D, start_position::Vector{T}, end_position::Vector{T}, initial_speed::T, T_horizon::Int64) where {T}
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

function linearized_dynamics(dyn::DoubleIntegrator2D, state::Vector{T}, control::Vector{T}) where {T}
    A = [1. 0. dyn.dt 0.; 0. 1. 0. dyn.dt; 0. 0. 1. 0.; 0. 0. 0. 1.]
    B = [0.5 * dyn.dt^2 0.; 0. 0.5 * dyn.dt^2; dyn.dt 0.; 0. dyn.dt]
    C = zeros(Float64, dyn.state_dim)
    A, B, C
end

@with_kw struct SingleIntegratorPolar2D{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64 = 2    # [x, y]
    ctrl_dim::Int64 = 2     # [θ, v]
    velocity_min::T = 0.
    velocity_max::T
    control_min::Vector{T}
    control_max::Vector{T}
end

# assuming control_lims are symmetric
SingleIntegratorPolar2D(dt::T, velocity_max::T, control_lim::Vector{T}) where {T} = SingleIntegratorPolar2D(dt=dt, velocity_max=velocity_max, control_min=-control_lim, control_max=control_lim)


function step(dyn::SingleIntegratorPolar2D, state, control)
    dt = dyn.dt
    θ, v = control
    dx = v * cos(θ) * dt
    dy = v * sin(θ) * dt
    state + [dx; dy]
end

get_position(dyn::SingleIntegratorPolar2D, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::SingleIntegratorPolar2D, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = control[end] .* [cos.(control[1]) sin.(control[1])]

function initial_straight_trajectory(dyn::SingleIntegratorPolar2D, start_position::Vector{T}, end_position::Vector{T}, initial_speed::T, T_horizon::Int64) where {T}
    x0, y0 = start_position
    xg, yg = end_position
    initial_heading = atan(yg - y0, xg - x0)
    initial_state = [x0, y0]
    states = Array{Float64}(undef, T_horizon + 1, dyn.state_dim)
    states[1,:] = initial_state
    controls = ones(Float64, T_horizon, dyn.ctrl_dim) * diagm([initial_heading, initial_speed])
    for t = 1:T_horizon
        states[t+1,:] = step(dyn, states[t,:], controls[t,:])
    end
    states, controls
end

function linearized_dynamics(dyn::SingleIntegratorPolar2D, state::Vector{T}, control::Vector{T}) where {T}
    A = ForwardDiff.jacobian(state -> step(dyn, state, control), state)
    B = ForwardDiff.jacobian(control -> step(dyn, state, control), control)
    C = step(dyn, state, control) - A * state - B * control
    A, B, C
end

@with_kw struct Unicycle{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64 = 3
    ctrl_dim::Int64 = 2
    velocity_min::T = 0.
    velocity_max::T
    control_min::Vector{T}
    control_max::Vector{T}
end

# assuming control_lims are symmetric
Unicycle(dt::T, velocity_max::T, control_lim::Vector{T}) where {T} = Unicycle(dt=dt, velocity_max=velocity_max, control_min=-control_lim, control_max=control_lim)


function step(dyn::Unicycle, state, control)
    dt = dyn.dt
    x, y, theta = state
    w, v = control
    eps = 1E-2
    w_ = abs(w) < eps ? 1.0 :  w
    dx = abs(w) < eps ? v * dt * cos(theta) : v / w_ * (sin(theta + w * dt) - sin(theta))
    dy = abs(w) < eps ? v * dt * sin(theta) : -v / w_ * (cos(theta + w * dt) - cos(theta))
    state + [dx; dy; w*dt]
end

get_position(dyn::Unicycle, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::Unicycle, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = control[..,2:2]

function initial_straight_trajectory(dyn::Unicycle, start_position::Vector{T}, end_position::Vector{T}, initial_speed::T, T_horizon::Int64) where {T}
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

function linearized_dynamics(dyn::Unicycle, state::Vector{T}, control::Vector{T}) where {T}
    A = ForwardDiff.jacobian(state -> step(dyn, state, control), state)
    B = ForwardDiff.jacobian(control -> step(dyn, state, control), control)
    C = step(dyn, state, control) - A * state - B * control
    A, B, C
end

@with_kw struct DynamicallyExtendedUnicycle{T} <: Dynamics where {T}
    dt::T
    state_dim::Int64 = 4
    ctrl_dim::Int64 = 2
    velocity_min::T = 0.
    velocity_max::T
    control_min::Vector{T}
    control_max::Vector{T}
end

DynamicallyExtendedUnicycle(dt::T, velocity_max::T, control_lim::Vector{T}) where {T} = DynamicallyExtendedUnicycle(dt=dt, velocity_max=velocity_max, control_min=-control_lim, control_max=control_lim)

function step(dyn::DynamicallyExtendedUnicycle, state, control)
    dt = dyn.dt
    x, y, theta, v = state
    w, a = control
    eps = 1E-2
    w_ = abs(w) < eps ? 1.0 :  w
    dx = abs(w) < eps ? (v * dt + 0.5 * a * dt^2) * cos(theta) : (w_ * (a * dt + v) * sin(theta + dt * w_) + a * cos(theta + dt * w_) - a * cos(theta) - v * w_ * sin(theta))/w_^2
    dy = abs(w) < eps ? (v * dt + 0.5 * a * dt^2) * sin(theta) : (-w_ * (a * dt + v) * cos(theta + dt * w_) + a * (sin(theta + dt * w_) - sin(theta)) + v * w_ * cos(theta))/w_^2

    state + [dx; dy; w * dt; a * dt]
end

get_position(dyn::DynamicallyExtendedUnicycle, state::VecOrMat{T}) where {T} = state[..,1:2]
get_velocity(dyn::DynamicallyExtendedUnicycle, state::VecOrMat{T}, control::VecOrMat{T}) where {T} = state[..,2:2]

function initial_straight_trajectory(dyn::DynamicallyExtendedUnicycle, start_position::Vector{T}, end_position::Vector{T}, initial_speed::T, T_horizon::Int64) where {T}
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

function linearized_dynamics(dyn::DynamicallyExtendedUnicycle, state::Vector{T}, control::Vector{T}) where {T}

    A = ForwardDiff.jacobian(state -> step(dyn, state, control), state)
    B = ForwardDiff.jacobian(control -> step(dyn, state, control), control)
    C = step(dyn, state, control) - A * state - B * control
    A, B, C
end
