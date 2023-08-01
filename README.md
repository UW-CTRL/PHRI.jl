# ProactiveHRI

[![Build Status](https://github.com/UW-CTRL/ProactiveHRI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UW-CTRL/ProactiveHRI.jl/actions/workflows/CI.yml?query=branch%3Amain)


# TODOs
- [x] investigate SingleIntegratorPolar2D
- [x] ibr function
- [x] mpc function
- [x] animation code
- [x] plot speed and control in plotting.jl
- [x] make summary plot (traj, speed, ctrl, collision, inconvenience) (see Jasper's code)
- [x] move things from planner.jl to planner_utils.jl
- [ ] parameter sweep
- [ ] make it a proper package
- [ ] ROS integration -- have human control human agent with xbox controller
- [x] Figure out how to remove the automatically generated compatibility branches.


![](https://github.com/UW-CTRL/ProactiveHRI.jl/blob/ibr_dev/animations/avoidance.gif)

# Installing as a Julia Package
To install ProactiveHRI.jl as a Julia package:
-  Go into the Julia REPL
-  Press ```]``` to enter package manager
-  Use the command ```add path/to/ProactiveHRI/repo/ProactiveHRI.jl```

# Using ProactiveHRI.jl
This is an example of how to use the ProactiveHRI.jl package to set up a planning problem between two agents (human and robot).

Define constants and parameters
```jl
dt = 0.1
velocity_max = 3.0
time_horizon = 25
markup = 1.05
collision_slack = 150.
trust_region_weight = 5.
inconvenience_weights = [1.; 1.; 0.01]
collision_radius = 1.
inconvenience_ratio = 0.2
solver = "ECOS"
```
Define the dynamics model used for both the robot and human agents.
```jl
human = Unicycle(dt, velocity_max, [1., 3.])
robot = DynamicallyExtendedUnicycle(dt, velocity_max, [1., 3.])
```
Set up the human Q, R and Qt matrices and define the human hyperparameters
```jl
Q = diagm([0.0; 0.0; 0.])
R = diagm([1.0; 0.0]) 
Qt = diagm([10.; 10.; 0.])

human_hps = PlannerHyperparameters(dynamics=human,
                             time_horizon=time_horizon,
                             Q=Q,
                             R=R,
                             Qt=Qt,
                             markup=markup,
                             collision_slack=collision_slack,
                             trust_region_weight=trust_region_weight,
                             inconvenience_weights=inconvenience_weights,
                             collision_radius=collision_radius,
                             inconvenience_ratio=inconvenience_ratio)
```
Do the same for the robot
```jl
Q = diagm([0.0; 0.0; 0.; 0.])
R = diagm([1.; 1.]) 
Qt = diagm([10.; 10.; 0.; 0.])

robot_hps = PlannerHyperparameters(dynamics=robot,
                             time_horizon=time_horizon,
                             Q=Q,
                             R=R,
                             Qt=Qt,
                             markup=markup,
                             collision_slack=collision_slack,
                             trust_region_weight=trust_region_weight,
                             inconvenience_weights=inconvenience_weights,
                             collision_radius=collision_radius,
                             inconvenience_ratio=inconvenience_ratio)
```
Define start and goal positions for both agents
```jl
robot_initial_state = [0.; 0.; 0.; 2.]
robot_goal_state = [10.; 0.; 0.; 2.]
human_initial_state = [10.; 0.; pi]
human_goal_state = [0.; 0.; pi]
```
Now the problem is ready to be set up as an interaction problem between the two agents using the ```InteractionPlanner``` function. It is also recommended to complete one solve once the Interaction Problem is defined
```jl
ip = InteractionPlanner(robot_hps, 
                        human_hps,
                        robot_initial_state,
                        human_initial_state,
                        robot_goal_state,
                        human_goal_state,
                        solver)

incon_problem, xs, us = @time solve(ip.ego_planner.incon, iterations=10, verbose=false, keep_history=false)
incon_problem, xs, us = @time solve(ip.other_planner.incon, iterations=10, verbose=false, keep_history=false)
```
Now, a solution using Iterated Best Response can be computed, with all the data being saved as a ```SaveData``` object. This data can later be used for plotting
```jl
saved_data_test, _, _, _, _ = ibr_save(ip, 3, "ego")
```
Now the problem is solved, to plot a summary plot:
```jl
plot_solve_solution(saved_data_test, scatter=false, show_speed=true, show_control=true)
```

![](https://github.com/UW-CTRL/ProactiveHRI.jl/blob/ibr_dev/figs/markup1dot1-headon.png)

