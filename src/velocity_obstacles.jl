using Interpolations
using MAT
using StaticArrays
using Parameters
using JuMP, HiGHS, ECOS

function relative_state(ego::DynamicallyExtendedUnicycle, state, other_position, other_velocity)
    # (xr, yr, θ_r, vr) : robot DynamicallyExtendedUnicycle state
    # (xh, yh) : other position
    # (vxh, vyh) : other velocity

    # (xrel, yrel) = R(θ_r) [xh - xr, yh - yr]
    # R = [cos(θ_r), sin(θ_r) ; -sin(θ_r), cos(θ_r)]
    # θ_rel - th - θ_r
    x_ego, y_ego, θ_ego, v_ego = state
    x_other, y_other = other_position
    vx_other, vy_other = other_velocity
    v_other = sqrt(vx_other^2 + vy_other^2)
    θ_other = atan(vy_other, vx_other)
    x_rel, y_rel = [cos(θ_ego) sin(θ_ego) ; -sin(θ_ego) cos(θ_ego)] * [x_other - x_ego; y_other - y_ego]
    θ_rel = θ_other - θ_ego
    return [x_rel, y_rel, θ_rel, v_other, v_ego]
end

function relative_dynamics(ego::DynamicallyExtendedUnicycle, state, other_position, other_velocity)
    # x_ego, y_ego, θ_ego, v_ego = state
    # x_other, y_other = other_position
    # vx_other, vy_other = other_velocity
    # v_other = sqrt(vx_other^2 + vy_other^2)
    # θ_other = atan(vy_other, vx_other)
    # x_rel, y_rel = [cos(θ_ego) sin(θ_ego) ; -sin(θ_ego) cos(θ_ego)] * [x_other - x_ego; y_other - y_ego]
    # θ_rel = θ_other - θ_ego
    x_rel, y_rel, θ_rel, v_other, v_ego = relative_state(ego, state, other_position, other_velocity)
    [v_other * cos(θ_rel) - v_ego; v_other * sin(θ_rel); 0.; 0.; 0.], [y_rel 0.; -x_rel 0.; -1. 0.; 0. 0.; 0. 1.]
end


function construct_VO_QP_base(ego::Dynamics, u_des::Vector{T}; γ=100.) where {T}
    # set up optimizer
    model = Model(HiGHS.Optimizer)
    model[:u] = @variable(model, u[1:2], base_name="u")
    model[:ϵ] = @variable(model, ϵ, base_name="ϵ")
    @objective(model, Min, sum((u - u_des)).^2 + γ * ϵ^2)
    model[:control_upper] = @constraint(model, u <= ego.control_max, base_name="control_upper")
    model[:control_lower] = @constraint(model, u >= ego.control_min, base_name="control_lower")
    model[:slack] = @constraint(model, ϵ >= 0, base_name="slack")
    model[:vo] = @constraint(model, u <= ego.control_max, base_name="vo") # temporary. Will be updated later when other agents' state is known
    model
end

function reactive_velocity_obstacles(vo_qp::Model, 
                                    ego::DynamicallyExtendedUnicycle, 
                                    state::Vector{T}, 
                                    desired_control::Vector{T},
                                    others_position::Vector{Vector{T}},
                                    others_velocity::Vector{Vector{T}};
                                    verbose=false,
                                    γ=100.,
                                    tol=1E-2) where {T}
    # for all other agents, find the "closest" agent wrt lowest HJ value function
    closest_other = 1
    V_min = 100.
    for (i,(p,v)) in enumerate(zip(others_position, others_velocity))
        rel_state = relative_state(ego, state, p, v)
        value = V(rel_state...)
        if value < V_min
            closest_other = i
            V_min = value
        end
    end

    closest_other_position = others_position[closest_other]
    closest_other_velocity = others_velocity[closest_other]
    closest_rel_state = relative_state(ego, state, closest_other_position, closest_other_velocity)
    # compute relative dynamics for closest agent
    f0, B = relative_dynamics(ego, state, closest_other_position, closest_other_velocity)

    # update objective with desired control
    @objective(vo_qp, Min, sum((vo_qp[:u] - desired_control).^2) + γ * vo_qp[:ϵ]^2)

    # update VO constraint
    delete_and_unregister(vo_qp, :vo)
    ∇V = gradient(V, closest_rel_state...)
    vo_qp[:vo] = @constraint(vo_qp, dot(∇V, f0 + B*vo_qp[:u]) >= -vo_qp[:ϵ])

    MOI.set(vo_qp, MOI.Silent(), !verbose)
    optimize!(vo_qp)


    if V_min > tol
        # if robot is not in danger of colliding, execute desired control
        desired_control, value.(vo_qp[:ϵ]), ∇V, f0, B, V_min
    else
        # if robor is in danger of colliding, react using VO 
        value.(vo_qp[:u]), value.(vo_qp[:ϵ]), ∇V, f0, B, V_min
    end

end
