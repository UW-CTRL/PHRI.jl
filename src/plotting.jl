using Plots
using Measures
using JuMP



function plot_solve_solution(problem::Problem; pos_xlims=[-1,8], pos_ylims=[-3, 3])
    l = @layout [a b c]
    width=1200
    height=300
    alpha0 = 0.2
    linewidth = 3

    xs = value.(problem.model[:x])
    us = value.(problem.model[:u])

    # plotting position trajectory
    goal_state = problem.opt_params.goal_state
    plot_traj = scatter(goal_state[1:1], goal_state[2:2], size=(width, height), xlabel="x position", ylabel="y position", title="Position", margin=5mm, color=:green, ylims=pos_ylims, xlims=pos_xlims, aspect_ratio=:equal, label="goal")
    plot!(xs[:,1], xs[:,2], color=:black, linewidth=linewidth, label=" ")
    scatter!(plot_traj, xs[:,1], xs[:,2], color=:black, label="")

    # plotting speed
    dynamics = problem.hps.dynamics
    N = problem.hps.time_horizon
    speed = get_speed(dynamics, xs, us)
    plot_speed = plot(speed, color=:blue, linewidth=linewidth, label="speed", size = (width, height), xlabel="time step", ylabel="Speed [m/s]", title="Speed", margin=5mm, ylim=[dynamics.velocity_min - 0.5, dynamics.velocity_max+0.5], legend=:bottomright)
    plot!(plot_speed, 1:N+1, dynamics.velocity_max * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:red, label="Max speed")
    plot!(plot_speed, 1:N+1, dynamics.velocity_min * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:green, label="Min speed")


    plot_ctrl = plot(us[:,1], color=:blue, linewidth=linewidth, label="u₁", size = (width, height), xlabel="time step", ylabel="control input", title="Controls", margin=5mm)
    plot!(plot_ctrl, us[:,2], color=:red, linewidth=linewidth+2, label="u₂")
    plot(plot_traj, plot_speed, plot_ctrl, layout = l)
end


function plot_solve_solution(problem::InteractionPlanner; pos_xlims=[-1,8], pos_ylims=[-3, 3])

    l = @layout [a b c] 
    width=1500
    height=500
    alpha0 = 0.2
    alpha_ideal = 0.4
    linewidth = 2
    markersize = 2
    markersize_large = 7
    ego_color = :blue
    other_color = :red



    ego_ideal = problem.ego_planner.ideal
    ego_incon = problem.ego_planner.incon
    other_ideal = problem.other_planner.ideal
    other_incon = problem.other_planner.incon

    ego_ideal_xs = value.(ego_ideal.model[:x])
    ego_ideal_us = value.(ego_ideal.model[:u])
    ego_incon_xs = vector_of_vectors_to_matrix(ego_incon.opt_params.previous_states)
    ego_incon_us = value.(ego_incon.model[:u])

    other_ideal_xs = value.(other_ideal.model[:x])
    other_ideal_us = value.(other_ideal.model[:u])
    other_incon_xs = vector_of_vectors_to_matrix(other_incon.opt_params.previous_states)
    other_incon_us = vector_of_vectors_to_matrix(other_incon.opt_params.previous_controls)

    ego_goal_state = ego_ideal.opt_params.goal_state
    other_goal_state = other_ideal.opt_params.goal_state

    # plotting position trajectory
    
    plot_traj = scatter(ego_goal_state[1:1], ego_goal_state[2:2], size=(width, height), xlabel="x position", ylabel="y position", title="Position", margin=10mm, marker=:star, markersize=markersize_large, color=ego_color, ylims=pos_ylims, xlims=pos_xlims, aspect_ratio=:equal, label="ego goal")
    scatter!(plot_traj, other_goal_state[1:1], other_goal_state[2:2], marker=:star, markersize=markersize_large, color=other_color, label="other goal")

    plot!(plot_traj, ego_ideal_xs[:,1], ego_ideal_xs[:,2], color=ego_color, linewidth=linewidth, label="", alpha=alpha_ideal)
    scatter!(plot_traj, ego_ideal_xs[:,1], ego_ideal_xs[:,2], color=ego_color, label="", alpha=alpha_ideal)
    plot!(plot_traj, ego_incon_xs[:,1], ego_incon_xs[:,2], color=ego_color, linewidth=linewidth, label="ego")
    scatter!(plot_traj, ego_incon_xs[:,1], ego_incon_xs[:,2], color=ego_color, label="")

    plot!(plot_traj, other_ideal_xs[:,1], other_ideal_xs[:,2], color=other_color, linewidth=linewidth, label="", alpha=alpha_ideal)
    scatter!(plot_traj, other_ideal_xs[:,1], other_ideal_xs[:,2], color=other_color, label="", alpha=alpha_ideal)
    plot!(plot_traj, other_incon_xs[:,1], other_incon_xs[:,2], color=other_color, linewidth=linewidth, label="other")
    scatter!(plot_traj, other_incon_xs[:,1], other_incon_xs[:,2], color=other_color, label="")

    # plotting speed

    ego_dynamics = problem.ego_planner.ideal.hps.dynamics
    other_dynamics = problem.other_planner.ideal.hps.dynamics
    N = problem.ego_planner.ideal.hps.time_horizon

    ego_ideal_speed = get_speed(ego_dynamics, ego_ideal_xs, ego_ideal_us)
    ego_incon_speed = get_speed(ego_dynamics, ego_incon_xs, ego_incon_us)
    other_ideal_speed = get_speed(other_dynamics, other_ideal_xs, other_ideal_us)
    other_incon_speed = get_speed(other_dynamics, other_incon_xs, other_incon_us)

    plot_speed = plot(size = (width, height), xlabel="time step", ylabel="Speed [m/s]", title="Speed", margin=5mm, ylim=[0, 3], legend=:bottomright)

    plot!(plot_speed, ego_ideal_speed, color=:blue, linewidth=linewidth, label="ego ideal speed", alpha=alpha_ideal)
    plot!(plot_speed, ego_incon_speed, color=:blue, linewidth=linewidth, label="ego incon speed")
    plot!(plot_speed, other_ideal_speed, color=:magenta, linewidth=linewidth, label="other ideal speed", alpha=alpha_ideal)
    plot!(plot_speed, other_incon_speed, color=:magenta, linewidth=linewidth, label="other incon speed")

    plot!(plot_speed, 1:N+1, ego_dynamics.velocity_max * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:red, label="Max speed")
    plot!(plot_speed, 1:N+1, ego_dynamics.velocity_min * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:green, label="Min speed")

    plot_ctrl = plot()
    
    plot!(plot_ctrl, ego_ideal_us[:,1], color=:blue, linewidth=linewidth, label="ego ideal u₁", size = (width, height), xlabel="time step", ylabel="control input", title="Controls", margin=10mm, alpha=alpha_ideal)
    plot!(plot_ctrl, ego_ideal_us[:,2], color=:red, linewidth=linewidth+2, label="ego ideal u₂", alpha=alpha_ideal)

    plot!(plot_ctrl, ego_incon_us[:,1], color=:blue, linewidth=linewidth, label="ego incon u₁")
    plot!(plot_ctrl, ego_incon_us[:,2], color=:red, linewidth=linewidth+2, label="ego incon u₂")

    plot!(plot_ctrl, other_ideal_us[:,1], color=:purple, linewidth=linewidth, label="other ideal u₁", alpha=alpha_ideal)
    plot!(plot_ctrl, other_ideal_us[:,2], color=:magenta, linewidth=linewidth+2, label="other ideal u₂", alpha=alpha_ideal)

    plot!(plot_ctrl, other_incon_us[:,1], color=:purple, linewidth=linewidth, label="other incon u₁")
    plot!(plot_ctrl, other_incon_us[:,2], color=:magenta, linewidth=linewidth+2, label="other incon u₂")

    plot(plot_traj, plot_ctrl, plot_speed, layout = l)
end

# summary plots dependent on iterations

function plot_solve_solution(problem::SaveData; pos_xlims=[-1,11], pos_ylims=[-6, 6], scatter=true::Bool, show_speed=true::Bool, show_control=true::Bool)

    l = @layout [a b c d e] 
    width=2000
    height=400
    alpha_ideal = 0.4
    linewidth = 2
    markersize = 2
    markersize_large = 7
    ego_color = :blue
    other_color = :red

    iterations = length(problem.previous_ips)

    alpha_ratio = 1 / (iterations + 1)
    N = problem.previous_ips[1].ego_planner.ideal.hps.time_horizon

    ego_goal_state = problem.previous_ips[1].ego_planner.ideal.opt_params.goal_state
    other_goal_state = problem.previous_ips[1].other_planner.ideal.opt_params.goal_state

    ego_dynamics = problem.previous_ips[1].ego_planner.ideal.hps.dynamics       # use first ip arbitrarily, any iteration will give the same values for these entries
    other_dynamics = problem.previous_ips[1].other_planner.ideal.hps.dynamics

    # plotting position trajectory

    plot_traj = plot(size=(height, height), xlabel="x position", ylabel="y position", title="Position", margin=10mm, ylims=pos_ylims, xlims=pos_xlims, aspect_ratio=:equal)
    scatter!(ego_goal_state[1:1], ego_goal_state[2:2], marker=:star, markersize=markersize_large, color=ego_color, label="ego goal")
    scatter!(plot_traj, other_goal_state[1:1], other_goal_state[2:2], marker=:star, markersize=markersize_large, color=other_color, label="other goal")

    plot!(plot_traj, value.(problem.previous_ips[1].ego_planner.ideal.model[:x])[:,1], value.(problem.previous_ips[1].ego_planner.ideal.model[:x])[:,2], color=ego_color, linewidth=linewidth, label="", alpha=alpha_ratio)

    plot!(plot_traj, value.(problem.previous_ips[1].other_planner.ideal.model[:x])[:,1], value.(problem.previous_ips[1].other_planner.ideal.model[:x])[:,2], color=other_color, linewidth=linewidth, label="", alpha=alpha_ratio)

    if scatter
        scatter!(plot_traj, value.(problem.previous_ips[1].ego_planner.ideal.model[:x])[:,1], value.(problem.previous_ips[1].ego_planner.ideal.model[:x])[:,2], color=ego_color, linewidth=linewidth, label="", alpha=alpha_ratio)

        scatter!(plot_traj, value.(problem.previous_ips[1].other_planner.ideal.model[:x])[:,1], value.(problem.previous_ips[1].other_planner.ideal.model[:x])[:,2], color=other_color, linewidth=linewidth, label="", alpha=alpha_ratio)
    end

    for i in 1:iterations
        plot!(plot_traj, vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_states)[:,1], vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_states)[:,2], color=ego_color, linewidth=linewidth, label="", alpha=(i * alpha_ratio))

        plot!(plot_traj, vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_states)[:,1], vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_states)[:,2], color=other_color, linewidth=linewidth, label="", alpha=(i * alpha_ratio))

        if scatter
            scatter!(plot_traj, vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_states)[:,1], vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_states)[:,2], color=ego_color, label="", alpha=(i * alpha_ratio))

            scatter!(plot_traj, vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_states)[:,1], vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_states)[:,2], color=other_color, label="", alpha=(i * alpha_ratio))
        end
    end

    if !scatter
        plot_traj[1][end - 1][:label] = "Ego Path"
        plot_traj[1][end][:label] = "Other Path"
    else
        plot_traj[1][end - 3][:label] = "Ego Path"
        plot_traj[1][end - 2][:label] = "Other Path"
    end
    # plotting speed/control

    # speed parameters
    max_speed = maximum([problem.previous_ips[1].ego_planner.ideal.hps.dynamics.velocity_max, problem.previous_ips[1].other_planner.ideal.hps.dynamics.velocity_max])
    ego_max_speed = problem.previous_ips[1].ego_planner.ideal.hps.dynamics.velocity_max
    other_max_speed = problem.previous_ips[1].other_planner.ideal.hps.dynamics.velocity_max

    # control parameters
    ego_ctrl_dim = problem.previous_ips[1].ego_planner.ideal.hps.dynamics.ctrl_dim
    other_ctrl_dim = problem.previous_ips[1].other_planner.ideal.hps.dynamics.ctrl_dim

    ego_max_ctrl = maximum(problem.previous_ips[1].ego_planner.ideal.hps.dynamics.control_max)
    ego_min_ctrl = minimum(problem.previous_ips[1].ego_planner.ideal.hps.dynamics.control_min)
    other_max_ctrl = maximum(problem.previous_ips[1].other_planner.ideal.hps.dynamics.control_max)
    other_min_ctrl = minimum(problem.previous_ips[1].other_planner.ideal.hps.dynamics.control_min)

    if show_speed & !show_control
        plot_speed_ego = plot(size=(height, height), xlabel="time step", ylabel="Speed [m/s]", title="Ego Speed", margin=10mm, ylim=[0, max_speed], legend=:bottomright)
        plot!(plot_speed_ego, 1:N+1, ego_max_speed * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:black, label="Max speed", linewidth=linewidth)
        plot!(plot_speed_ego, 1:N+1, 0 * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:green, label="Min speed", linewidth=linewidth)

        plot_speed_other = plot(size=(height, height), xlabel="time step", ylabel="Speed [m/s]", title="Other Speed", margin=10mm, ylim=[0, max_speed], legend=:bottomright)
        plot!(plot_speed_other, 1:N+1, other_max_speed * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:black, label="Max speed", linewidth=linewidth)
        plot!(plot_speed_other, 1:N+1, 0 * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:green, label="Min speed", linewidth=linewidth)

        for i in 1:iterations
            ego_speed = get_speed(ego_dynamics, vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_states), vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_controls))
            other_speed = get_speed(other_dynamics, vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_states), vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_controls))

            plot!(plot_speed_ego, 1:N, ego_speed[1:N], alpha=(i * alpha_ratio), color=ego_color, label="", linewidth=linewidth)
            plot!(plot_speed_other, 1:N, other_speed[1:N], alpha=(i * alpha_ratio), color=other_color, label="", linewidth=linewidth)
        end

        plot_speed_ego[1][end][:label] = "Ego Speed"
        plot_speed_other[1][end][:label] = "Other Speed"

    elseif show_control & !show_speed
        plot_ctrl_ego = plot(size=(height, height), xlabel="time step", ylabel="input magnitude", title="Ego Control", margin=10mm)
        plot_ctrl_other = plot(size=(height, height), xlabel="time step", ylabel="input magnitude", title="Other Control", margin=10mm)

        plot!(plot_ctrl_ego, 1:N, ego_max_ctrl * ones(Float64, N), linestyle=:dash, linewith=linewidth, color=:green, label="Control Limits")
        plot!(plot_ctrl_ego, 1:N, ego_min_ctrl * ones(Float64, N), linestyle=:dash, linewith=linewidth, color=:green, label="")
        plot!(plot_ctrl_other, 1:N, other_max_ctrl * ones(Float64, N), linestyle=:dash, linewith=linewidth, color=:green, label="Control Limits")
        plot!(plot_ctrl_other, 1:N, other_min_ctrl * ones(Float64, N), linestyle=:dash, linewith=linewidth, color=:green, label="")

        # nested for loops :|
        for i in 1:iterations
            # ego plot
            for j in 1:ego_ctrl_dim
                plot!(plot_ctrl_ego, 1:N, vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_controls)[:, j], label="", color=RGB(1 - (1 / ego_ctrl_dim) * j, 0., (1 / ego_ctrl_dim) * j), linewidth=linewidth, alpha=(i * alpha_ratio))
            end

            # other plot
            for k in 1:other_ctrl_dim
                plot!(plot_ctrl_other, 1:N, vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_controls)[:, k], label="", color=RGB((1 / other_ctrl_dim) * k, 1 - (1 / other_ctrl_dim) * k, 0.), linewidth=linewidth, alpha=(i * alpha_ratio))
            end
        end

        for l in 1:ego_ctrl_dim
            plot_ctrl_ego[1][end - (l - 1)][:label] = "u$(ego_ctrl_dim - l + 1)"
        end

        for m in 1:ego_ctrl_dim
            plot_ctrl_other[1][end - (m - 1)][:label] = "u$(ego_ctrl_dim - m + 1)"
        end
    elseif show_speed & show_control
        plot_speed = plot(size=(height, height), xlabel="time step", ylabel="Speed [m/s]", title="Speed", margin=10mm, ylim=[0, max_speed], legend=:bottomright)
        plot_ctrl = plot(size=(height, height), xlabel="time step", ylabel="input magnitude", title="Control", margin=10mm)

        # speed plotting
        plot!(plot_speed, 1:N+1, maximum([ego_max_speed, other_max_speed]) * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:black, label="Max speed", linewidth=linewidth)
        plot!(plot_speed, 1:N+1, 0 * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:green, label="Min speed", linewidth=linewidth)

        for i in 1:iterations
            ego_speed = get_speed(ego_dynamics, vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_states), vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_controls))
            other_speed = get_speed(other_dynamics, vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_states), vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_controls))

            plot!(plot_speed, 1:N, ego_speed[1:N], alpha=(i * alpha_ratio), color=ego_color, label="", linewidth=linewidth)
            plot!(plot_speed, 1:N, other_speed[1:N], alpha=(i * alpha_ratio), color=other_color, label="", linewidth=linewidth)
        end
        plot_speed[1][end - 1][:label] = "Ego Speed"
        plot_speed[1][end][:label] = "Other Speed"

        # ctrl plotting
        plot!(plot_ctrl, 1:N, maximum([ego_max_ctrl, other_max_ctrl]) * ones(Float64, N), linestyle=:dash, linewith=linewidth, color=:green, label="Control Limits")
        plot!(plot_ctrl, 1:N, minimum([ego_min_ctrl, other_min_ctrl]) * ones(Float64, N), linestyle=:dash, linewith=linewidth, color=:green, label="")

        for i in 1:iterations
            # ego plot
            for j in 1:ego_ctrl_dim
                plot!(plot_ctrl, 1:N, vector_of_vectors_to_matrix(problem.previous_ips[i].ego_planner.incon.opt_params.previous_controls)[:, j], label="", color=RGB(1 - (1 / ego_ctrl_dim) * j, 0., (1 / ego_ctrl_dim) * j), linewidth=linewidth, alpha=(i * alpha_ratio))
            end

            # other plot
            for k in 1:other_ctrl_dim
                plot!(plot_ctrl, 1:N, vector_of_vectors_to_matrix(problem.previous_ips[i].other_planner.incon.opt_params.previous_controls)[:, k], label="", color=RGB((1 / other_ctrl_dim) * k, 1 - (1 / other_ctrl_dim) * k, 0.), linewidth=linewidth, alpha=(i * alpha_ratio))
            end
        end

        for l in 1:ego_ctrl_dim
            plot_ctrl[1][end - (l + 1)][:label] = "Ego u$(ego_ctrl_dim - l + 1)"
        end

        for m in 1:ego_ctrl_dim
            plot_ctrl[1][end - (m - 1)][:label] = "Other u$(other_ctrl_dim - m + 1)"
        end
    end

    # plotting slack violation over iterations

    slack_violation = Vector{Float64}(undef, iterations)

    for i in 1:iterations
        slack_violation[i] = value(problem.previous_ips[i].ego_planner.incon.model[:ϵ])
    end

    plot_slack_violation = plot(size=(height, height), xlabel="Iteration", ylabel="ϵ (slack value)", title="Slack (collision) Violation", margin=10mm)

    plot!(plot_slack_violation, 1:iterations, slack_violation, color=:black, label="Slack")

    # plotting inconvenience value over iterations

    incon_budget = problem.previous_ips[1].ego_planner.incon.hps.inconvenience_ratio
    inconvenience_ego = Vector{Float64}(undef, iterations)
    inconvenience_other = Vector{Float64}(undef, iterations)

    ideal_incon_ego = compute_convenience_value(ego_dynamics, matrix_to_vector_of_vectors(value.(problem.previous_ips[1].ego_planner.ideal.model[:x])), matrix_to_vector_of_vectors(value.(problem.previous_ips[1].ego_planner.ideal.model[:u])), ego_goal_state, problem.previous_ips[1].ego_planner.incon.hps.inconvenience_weights)

    ideal_incon_other = compute_convenience_value(other_dynamics, matrix_to_vector_of_vectors(value.(problem.previous_ips[1].other_planner.ideal.model[:x])), matrix_to_vector_of_vectors(value.(problem.previous_ips[1].other_planner.ideal.model[:u])), other_goal_state, problem.previous_ips[1].other_planner.incon.hps.inconvenience_weights)

    for i in 1:iterations
        inconvenience_ego[i] = compute_convenience_value(ego_dynamics, problem.previous_ips[i].ego_planner.incon.opt_params.previous_states, problem.previous_ips[i].ego_planner.incon.opt_params.previous_controls, ego_goal_state, problem.previous_ips[1].ego_planner.incon.hps.inconvenience_weights)
        
        inconvenience_other[i] = compute_convenience_value(other_dynamics, problem.previous_ips[i].other_planner.incon.opt_params.previous_states, problem.previous_ips[i].other_planner.incon.opt_params.previous_controls, other_goal_state, problem.previous_ips[1].other_planner.incon.hps.inconvenience_weights)
    end

    inconvenience_ego ./= ideal_incon_ego
    inconvenience_other ./= ideal_incon_other 

    plot_incon = plot(size=(height, height), xlabel="Iteration", ylabel="Inconvenience", title="Agent Inconvenience", margin=10mm)
    plot!(plot_incon, 1:iterations, ones(iterations), linestyle=:dash, linewith=linewidth, color=:green, label="Ideal Incon")
    plot!(plot_incon, 1:iterations, ones(iterations) .+ incon_budget, linestyle=:dash, linewith=linewidth, color=:black, label="Incon Budget")
    plot!(plot_incon, 1:iterations, inconvenience_ego, color=ego_color, linewidth=linewidth, label="Ego Incon")
    plot!(plot_incon, 1:iterations, inconvenience_other, color=other_color, linewidth=linewidth, label="Other Incon")


    if show_speed & !show_control
        plot(plot_traj, plot_speed_ego, plot_speed_other, plot_slack_violation, plot_incon, layout=l, size=(width, height))
    elseif show_control & !show_speed
        plot(plot_traj, plot_ctrl_ego, plot_ctrl_other, plot_slack_violation, plot_incon, layout=l, size=(width, height))
    elseif show_speed & show_control
        plot(plot_traj, plot_speed, plot_ctrl, plot_slack_violation, plot_incon, layout=l, size=(width, height))
    end
end

# plotting from the SimData object
function plot_solve_solution(problem::SimData; pos_xlims=[-1,11], pos_ylims=[-6, 6])

    l = @layout [a b c] 
    width=1500
    height=500
    alpha0 = 0.2
    alpha_ideal = 0.4
    linewidth = 2
    markersize = 2
    markersize_large = 7
    ego_color = :blue
    other_color = :red

    ego_xs = problem.ego_states
    ego_us = problem.ego_controls

    other_xs = problem.other_states
    other_us = problem.other_controls

    ego_goal_state = problem.sim_params.ego_planner_params.opt_params.goal_state
    other_goal_state = problem.sim_params.other_planner_params.opt_params.goal_state

    # plotting position trajectory
    
    plot_traj = scatter(ego_goal_state[1:1], ego_goal_state[2:2], size=(width, height), xlabel="x position", ylabel="y position", title="Position", margin=10mm, marker=:star, markersize=markersize_large, color=ego_color, ylims=pos_ylims, xlims=pos_xlims, aspect_ratio=:equal, label="ego goal")
    scatter!(plot_traj, other_goal_state[1:1], other_goal_state[2:2], marker=:star, markersize=markersize_large, color=other_color, label="other goal")

    plot!(plot_traj, ego_xs[:,1], ego_xs[:,2], color=ego_color, linewidth=linewidth, label="ego")
    scatter!(plot_traj, ego_xs[:,1], ego_xs[:,2], color=ego_color, label="")

    plot!(plot_traj, other_xs[:,1], other_xs[:,2], color=other_color, linewidth=linewidth, label="other")
    scatter!(plot_traj, other_xs[:,1], other_xs[:,2], color=other_color, label="")

    # plotting speed

    ego_dynamics = problem.sim_params.ego_planner_params.hps.dynamics
    other_dynamics = problem.sim_params.other_planner_params.hps.dynamics
    N = problem.sim_params.ego_planner_params.hps.time_horizon

    ego_speed = get_speed(ego_dynamics, ego_xs, ego_us)
    other_speed = get_speed(other_dynamics, other_xs, other_us)

    plot_speed = plot(size = (width, height), xlabel="time step", ylabel="Speed [m/s]", title="Speed", margin=5mm, ylim=[0, 3], legend=:bottomright)

    plot!(plot_speed, ego_speed, color=:blue, linewidth=linewidth, label="ego incon speed")
    plot!(plot_speed, other_speed, color=:magenta, linewidth=linewidth, label="other incon speed")

    plot!(plot_speed, 1:N+1, ego_dynamics.velocity_max * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:red, label="Max speed")
    plot!(plot_speed, 1:N+1, ego_dynamics.velocity_min * ones(Float64, N+1), linestyle=:dash, linewith=linewidth, color=:green, label="Min speed")

    plot_ctrl = plot(size = (width, height), xlabel="time step", ylabel="control input", title="Controls", margin=10mm, alpha=alpha_ideal)

    plot!(plot_ctrl, ego_us[:,1], color=:blue, linewidth=linewidth, label="ego incon u₁")
    plot!(plot_ctrl, ego_us[:,2], color=:red, linewidth=linewidth+2, label="ego incon u₂")

    plot!(plot_ctrl, other_us[:,1], color=:purple, linewidth=linewidth, label="other incon u₁")
    plot!(plot_ctrl, other_us[:,2], color=:magenta, linewidth=linewidth+2, label="other incon u₂")

    plot(plot_traj, plot_ctrl, plot_speed, layout = l)
end


function animation(ip::InteractionPlanner; pos_xlims=[-1, 8], pos_ylims=[-3, 3], save_name="none")
    a = Animation()

    linewidth = 3
    alpha_ideal = 0.2
    ego_color = :blue
    other_color = :red

    ego_ideal = ip.ego_planner.ideal
    ego_incon = ip.ego_planner.incon
    other_ideal = ip.other_planner.ideal
    other_incon = ip.other_planner.incon

    ego_ideal_xs = value.(ego_ideal.model[:x])
    ego_incon_xs = value.(ego_incon.model[:x])
    other_ideal_xs = value.(other_ideal.model[:x])
    other_incon_xs = value.(other_incon.model[:x])

    plt = plot(xlim=pos_xlims, ylim=pos_ylims, xlabel="x position", ylabel="y position", title="Position Animation", arrow=true)
    plot!(plt, ego_incon_xs[1:1,1], ego_incon_xs[1:1,2], color=ego_color, linewidth=linewidth, lab="Robot")
    plot!(plt, other_incon_xs[1:1,1], other_incon_xs[1:1,2], color=other_color, linewidth=linewidth, lab="Human")

    for i in 1:ip.ego_planner.ideal.hps.time_horizon
        # plot!(plt, ego_ideal_xs[1:i,1], ego_ideal_xs[1:i,2], color=:purple, linewidth=linewidth, lab="", alpha=alpha_ideal)
        plot!(plt, ego_incon_xs[1:i,1], ego_incon_xs[1:i,2], color=ego_color, linewidth=linewidth, lab="")
        # plot!(plt, other_ideal_xs[1:i,1], other_ideal_xs[1:i,2], color=:magenta, linewidth=linewidth, lab="", alpha=alpha_ideal)
        plot!(plt, other_incon_xs[1:i,1], other_incon_xs[1:i,2], color=other_color, linewidth=linewidth, lab="")
        frame(a, plt)
    end

    if save_name != "none"
        gif(a, "../animations/$save_name.gif", fps = 15) 
    end 

    return gif(a, fps=60)
end

function avoidance_animation(ip::InteractionPlanner; pos_xlims=[-1, 8], pos_ylims=[-3, 3], save_name="none")
    a = Animation()

    linewidth = 3
    alpha_ideal = 0.2
    ego_color = :blue
    other_color = :red

    ego_ideal = ip.ego_planner.ideal
    ego_incon = ip.ego_planner.incon
    other_ideal = ip.other_planner.ideal
    other_incon = ip.other_planner.incon

    ego_ideal_xs = value.(ego_ideal.model[:x])
    ego_incon_xs = vector_of_vectors_to_matrix(ego_incon.opt_params.previous_states)
    other_ideal_xs = value.(other_ideal.model[:x])
    other_incon_xs = vector_of_vectors_to_matrix(other_incon.opt_params.previous_states)

    # plt = plot(xlim=pos_xlims, ylim=pos_ylims, xlabel="x position", ylabel="y position", title="Position Animation", arrow=true)
    # plot!(plt, ego_incon_xs[1:1,1], ego_incon_xs[1:1,2], color=ego_color, linewidth=linewidth, lab="Robot")
    # plot!(plt, other_incon_xs[1:1,1], other_incon_xs[1:1,2], color=other_color, linewidth=linewidth, lab="Human")


    print(typeof(ip.ego_planner.ideal.hps.time_horizon))

    for i in 1:ip.ego_planner.ideal.hps.time_horizon
        # plot!(plt, ego_ideal_xs[1:i,1], ego_ideal_xs[1:i,2], color=:purple, linewidth=linewidth, lab="", alpha=alpha_ideal)
        plt = plot(ego_incon_xs[1:i,1], ego_incon_xs[1:i,2], color=ego_color, linewidth=linewidth, lab="", xlim=pos_xlims, ylim=pos_ylims)
        plot!(plt, [ego_incon_xs[i,1]], [ego_incon_xs[i,2]], marker=:circle, color=:red, markersize=25, lab="", alpha=i/250)
        # plot!(plt, other_ideal_xs[1:i,1], other_ideal_xs[1:i,2], color=:magenta, linewidth=linewidth, lab="", alpha=alpha_ideal)
        plot!(plt, other_incon_xs[1:i,1], other_incon_xs[1:i,2], color=other_color, linewidth=linewidth, lab="")
        plot!(plt, [other_incon_xs[i,1]], [other_incon_xs[i,2]], marker=:circle, color=:red, markersize=25, lab="", alpha=i/250)

        frame(a, plt)
    end

    if save_name != "none"
        gif(a, "../animations/$save_name.gif", fps = 15) 
    end 

    return gif(a, fps=60)
end 

# animate function for MPC sim
function animation(ego_path::Matrix{Float64}, other_path::Matrix{Float64}; pos_xlims=[-1, 8], pos_ylims=[-3, 3], save_name="none")
    a = Animation()

    linewidth = 3
    alpha_ideal = 0.2
    ego_color = :blue
    other_color = :red

    ego_xs = ego_path
    other_xs = other_path

    plt = plot(xlim=pos_xlims, ylim=pos_ylims, xlabel="x position", ylabel="y position", title="Position Animation", arrow=true)


    for i in 1:length(ego_xs[:, 1]) - 1
        # plot!(plt, ego_ideal_xs[1:i,1], ego_ideal_xs[1:i,2], color=:purple, linewidth=linewidth, lab="", alpha=alpha_ideal)
        plt = plot(ego_xs[1:i,1], ego_xs[1:i,2], color=ego_color, linewidth=linewidth, lab="", xlim=pos_xlims, ylim=pos_ylims, xlabel="x position", ylabel="y position", title="Position Animation")
        # plot!(plt, other_ideal_xs[1:i,1], other_ideal_xs[1:i,2], color=:magenta, linewidth=linewidth, lab="", alpha=alpha_ideal)
        plot!(plt, other_xs[1:i,1], other_xs[1:i,2], color=other_color, linewidth=linewidth, lab="")
        frame(a, plt)
    end

    if save_name != "none"
        gif(a, "../animations/$save_name.gif", fps = 15) 
    end 

    return gif(a, fps=60)
end