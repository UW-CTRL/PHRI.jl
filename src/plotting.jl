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
