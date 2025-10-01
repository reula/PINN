# -------------------------------------------------------------------
# Physics-Informed Neural Network (PINN) for Solving 2D Poisson's Equation
#
# This script demonstrates how to use a neural network to solve the
# Poisson equation with Dirichlet boundary conditions in a unit square.
#
# The PINN approach embeds the PDE and boundary conditions into the loss
# function, so the neural network learns a solution that satisfies both.
# -------------------------------------------------------------------

using ComponentArrays
using Distributions
using GLMakie
using Lux
using LuxCUDA
using OptimizationOptimJL
using Random
using UnPack
using Zygote

# -------------------------------------------------------------------
# Neural Network Construction
# -------------------------------------------------------------------
"""
    create_neural_network(config)

Creates a fully-connected neural network (MLP) based on the configuration.
Returns the network, its parameters, and its state.
"""
function create_neural_network(config)
    @unpack N_input, N_neurons, N_layers, N_output = config

    # Initialize random number generator for reproducibility
    rng = Random.default_rng()
    Random.TaskLocalRNG()
    Random.seed!(rng)

    # Build the neural network: input layer, hidden layers, output layer
    NN = Chain(
        Dense(N_input, N_neurons, tanh),
        [Dense(N_neurons, N_neurons, tanh) for _ in 1:N_layers]...,
        Dense(N_neurons, N_output)
    )

    # Initialize parameters and state (state is unused but required)
    Θ, st = Lux.setup(rng, NN)

    # Move parameters to GPU and wrap for optimization
    Θ = Θ |> ComponentArray |> gpu_device() .|> Float64

    return NN, Θ, st
end

# -------------------------------------------------------------------
# Input Data Generation
# -------------------------------------------------------------------
"""
    generate_input(config)

Generates random (x, y) points in the domain as input for the PINN.
"""
function generate_input(config)
    @unpack N_points, xmin, xmax, ymin, ymax = config

    x = rand(Uniform(xmin, xmax), (1, N_points))
    y = rand(Uniform(ymin, ymax), (1, N_points))

    # Stack and move to GPU
    input = vcat(x, y) |> gpu_device() .|> Float64
    return input
end

# -------------------------------------------------------------------
# PINN Solution Representation
# -------------------------------------------------------------------
"""
    calculate_f(x, y, NN, Θ, st)

Returns the PINN's prediction for u(x, y), enforcing Dirichlet boundary conditions.
"""
function calculate_f(x, y, NN, Θ, st)
    # Hard enforcement of boundary conditions:
    # u(x, 0) = x^2, u(0, y) = y^2, u(1, y) = 1 + y^2, u(x, 1) = 1 + x^2
    return x .^ 2 .+ y .^ 2 .+ x .* y .* (x .- 1) .* (y .- 1) .* NN(vcat(x, y), Θ, st)[1]
end

# -------------------------------------------------------------------
# Numerical Derivatives (Finite Differences)
# -------------------------------------------------------------------
"""
    calculate_derivatives(x, y, NN, Θ, st)

Computes u, ∂²u/∂x², and ∂²u/∂y² at (x, y) using central finite differences.
"""
function calculate_derivatives(x, y, NN, Θ, st)
    ϵ = ∜(eps())  # Optimal step for second derivatives

    f = calculate_f(x, y, NN, Θ, st)
    f_xplus  = calculate_f(x .+ ϵ, y, NN, Θ, st)
    f_xminus = calculate_f(x .- ϵ, y, NN, Θ, st)
    f_yplus  = calculate_f(x, y .+ ϵ, NN, Θ, st)
    f_yminus = calculate_f(x, y .- ϵ, NN, Θ, st)

    ∂2f_∂x2 = (f_xplus .- 2 * f .+ f_xminus) / ϵ^2
    ∂2f_∂y2 = (f_yplus .- 2 * f .+ f_yminus) / ϵ^2

    return f, ∂2f_∂x2, ∂2f_∂y2
end

# -------------------------------------------------------------------
# Source Term and PDE Residual
# -------------------------------------------------------------------
"""
    calculate_source_term(x, y)

Returns the source term f(x, y) for the Poisson equation.
"""
calculate_source_term(x, y) = 4  # f(x, y) = 4 everywhere

"""
    poisson_equation(∂2u_∂x2, ∂2u_∂y2, source_term)

Returns the residual of the Poisson equation at each point.
"""
poisson_equation(∂2u_∂x2, ∂2u_∂y2, source_term) = ∂2u_∂x2 .+ ∂2u_∂y2 .- source_term

# -------------------------------------------------------------------
# Loss Function
# -------------------------------------------------------------------
"""
    loss_function(input, NN, Θ, st)

Computes the mean squared error of the PDE residual over all input points.
"""
function loss_function(input, NN, Θ, st)
    x, y = input[1:1, :], input[2:2, :]
    _, ∂2f_∂x2, ∂2f_∂y2 = calculate_derivatives(x, y, NN, Θ, st)
    source_term = calculate_source_term(x, y)
    eq = poisson_equation(∂2f_∂x2, ∂2f_∂y2, source_term)
    loss = sum(abs2, eq) / length(eq)
    return loss
end

# -------------------------------------------------------------------
# Optimization Callback
# -------------------------------------------------------------------
"""
    callback(p, l, losses)

Stores and prints the loss at each optimization step.
"""
function callback(p, l, losses)
    push!(losses, l)
    println("Current loss: ", l)
    return false
end

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
config = Dict(
    :N_input => 2,          # Number of input variables (x, y)
    :N_neurons => 30,       # Neurons per hidden layer
    :N_layers => 2,         # Number of hidden layers
    :N_output => 1,         # Output dimension (u)
    :N_points => 1000,      # Training points
    :xmin => 0.0,           # Domain: x in [xmin, xmax]
    :xmax => 1.0,
    :ymin => 0.0,           # Domain: y in [ymin, ymax]
    :ymax => 1.0,
    :optimizer => BFGS(),   # Optimization algorithm
    :maxiters => 1000,      # Maximum optimization steps
)

# -------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------
losses = Float64[]  # Store loss history

NN, Θ, st = create_neural_network(config)
input = generate_input(config)

# Define the optimization problem for the PINN
optf = OptimizationFunction((Θ, input) -> loss_function(input, NN, Θ, st), AutoZygote())
optprob = OptimizationProblem(optf, Θ, input)
optresult = solve(
    optprob,
    callback = (p, l) -> callback(p, l, losses),
    config[:optimizer],
    maxiters = config[:maxiters],
)

# Plot loss curve
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], title = "Loss vs Iterations", xlabel = "Iterations", ylabel = "Loss", yscale = log10)
lines!(ax, losses)
display(fig)

# -------------------------------------------------------------------
# Evaluation and Visualization
# -------------------------------------------------------------------
# Move optimized parameters back to CPU for evaluation
Θ = optresult.u |> cpu_device()

"""
    analytical_solution(x, y)

Returns the exact solution for the Poisson equation with these boundary conditions.
"""
analytical_solution(x, y) = x .^ 2 .+ y .^ 2

# Create a grid for evaluation
x = reshape([x for x in range(0, 1, 50) for y in range(0, 1, 50)], 1, :)
y = reshape([y for x in range(0, 1, 50) for y in range(0, 1, 50)], 1, :)

# Evaluate both solutions
f_analytical = analytical_solution.(x, y)
f_pinn = calculate_f(x, y, NN, Θ, st)

# Compute error metrics
mean_abs_error = sum(abs.(f_analytical .- f_pinn)) / length(f_analytical)
mask = f_analytical .!= 0
mean_rel_error = sum(abs.((f_analytical[mask] .- f_pinn[mask]) ./ f_analytical[mask]))
max_error = maximum(abs.(f_analytical .- f_pinn))

println("Mean absolute error: ", mean_abs_error)
println("Mean relative error: ", mean_rel_error)
println("Max absolute error: ", max_error)

# Prepare data for contour plots
x1 = reshape(x, 50, 50)[1, :]
y1 = reshape(y, 50, 50)[:, 1]
f_analytical1 = reshape(f_analytical, 50, 50)
f_pinn1 = reshape(f_pinn, 50, 50)
abs_error = abs.(f_analytical1 .- f_pinn1)
rel_error = abs_error ./ abs.(f_analytical1)

# Plot analytical solution, PINN solution, and errors
fig = Figure()
ax1 = GLMakie.Axis(fig[1, 1], title = "Analytical Solution", xlabel = "x", ylabel = "y")
cntr_an = contourf!(ax1, x1, y1, f_analytical1, colormap = :plasma)
Colorbar(fig[1,2], cntr_an)
ax2 = GLMakie.Axis(fig[1, 3], title = "PINN Solution", xlabel = "x", ylabel = "y")
cntr_pinn = contourf!(ax2, x1, y1, f_pinn1, colormap = :plasma)
Colorbar(fig[1,4], cntr_pinn)
ax3 = GLMakie.Axis(fig[2, 1], title = "Absolute Error", xlabel = "x", ylabel = "y")
cntr_abs = contourf!(ax3, x1, y1, abs_error, colormap = :plasma)
Colorbar(fig[2, 2], cntr_abs)
ax4 = GLMakie.Axis(fig[2, 3], title = "Relative Error", xlabel = "x", ylabel = "y")
cntr_rel = contourf!(ax4, x1, y1, rel_error, colormap = :plasma)
Colorbar(fig[2, 4], cntr_rel)
display(fig)