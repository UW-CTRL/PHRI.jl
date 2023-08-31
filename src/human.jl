using ForwardDiff
using Plots

# social forces policy
get_unit_vector(vec) = vec / norm(vec)
barrier(r_ab, other_speed, other_position, dt) = 0.5 * sqrt((norm(r_ab) + norm(r_ab - other_speed * dt * other_position))^2 - (other_speed * dt)^2)

pedestrian_potential_function(barrier, a, b) =  a * exp(-barrier / b)

function field_of_view(ψ, c, direction, force)
    if dot(direction, force) >= norm(force) * cos(ψ)
        return 1
    else
        return c
    end
end

function social_forces(ego, state, goal_position, others, desired_velocity; p=2., q=2., τ=2., ψ=pi/6, c=0.3)
    dt = ego.dt
    # goal force
    position = get_position(ego, state)
    velocity = get_velocity(ego, state, zeros(1, ego.ctrl_dim))
    goal_direction = get_unit_vector(goal_position - position)
    force = (desired_velocity .* goal_direction - velocity) / τ

    # pedestrian force
    for (dyn_o, o_state) in others
        o_position = get_position(dyn_o, o_state)
        # o_velocity = get_velocity(dyn_o, o_state, [0.; 0.])
        o_speed = get_speed(dyn_o, o_state, [0.; 0.])[1]
        V_b_x(x) = pedestrian_potential_function(barrier(x - o_position, o_speed, o_position, dt), p, q)
        ped_force = -ForwardDiff.gradient(V_b_x, position)
        fov_weight = field_of_view(ψ, c, position, -ped_force)
        force += fov_weight * ped_force
    end
    if norm(force) > norm(ego.control_max)
        force /= norm(force) * norm(ego.control_max)
    end
    force
end

# # example usage
# state = [1.; 0.; 1.; 0.]
# goal_position = [10.; 1.] 
# others = [[ego, [2; .0; -.0; 0.]]]
# desired_velocity = 2.
# forces = social_forces(ego, state, goal_position, others, desired_velocity, p=2., ψ=pi/6, c=0.3)

# markersize = 15
# scatter(state[1:1], state[2:2], markersize=markersize, label="ego", aspect_ratio=:equal)
# for (_, o) in others
#     scatter!(o[1:1], o[2:2], markersize=markersize, label="other")
# end
# scatter!(goal_position[1:1], goal_position[2:2], markersize=markersize, label="goal")
# plot!([state[1], state[1] + forces[1]], [state[2], state[2] + forces[2]], label="force", linewidth=5, color="black")
