function Sim(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64)
    # Given the IP problem setup of the ego agent and other agent
    ego_initial_control = ego_ip.ego_planner.incon.opt_params.previous_controls[1]
    other_initial_control = other_ip.ego_planner.incon.opt_params.previous_controls[1]
    for i in 1:sim_horizon
        next_ego_state = step(ego_ip.ego_planner.)

    # Uses MPC function to simulate to a given time horizon
end