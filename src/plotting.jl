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

function special_animation(ip::InteractionPlanner; pos_xlims=[-1, 8], pos_ylims=[-3, 3], save_name="none")
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


    for i in 1:length(ego_xs[:, 1])
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