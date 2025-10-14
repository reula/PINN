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
    NN = create_chain(config)

    # Initialize parameters and state (state is unused but required)
    Θ, st = Lux.setup(rng, NN)

    # Move parameters to GPU and wrap for optimization
    Θ = Θ |> ComponentArray |> gpu_device() .|> Float64

    return NN, Θ, st
end

function create_chain(config)
    @unpack N_input, N_neurons, N_layers, N_output = config

    # Build the neural network: input layer, hidden layers, output layer
    NN = Lux.Chain(
        Lux.Dense(N_input, N_neurons, tanh),
        [Lux.Dense(N_neurons, N_neurons, tanh) for _ in 1:N_layers]...,
        Lux.Dense(N_neurons, N_output)
    )

    return NN
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
    #input = vcat(x, y) |> gpu_device() .|> Float32

    return input
end

function generate_input(config, N_points)
    @unpack xmin, xmax, ymin, ymax = config

    x = rand(Uniform(xmin, xmax), (1, N_points))
    y = rand(Uniform(ymin, ymax), (1, N_points))

    # Stack and move to GPU
    
    input = vcat(x, y) |> gpu_device() .|> Float64
    #input = vcat(x, y) |> gpu_device() .|> Float32

    return input
end
# -------------------------------------------------------------------
# Muestreo (x,t)
# -------------------------------------------------------------------
function generate_input_x_t(config)
    @unpack N_points, xmin, xmax, tmin, tmax = config
    x = rand(Uniform(xmin, xmax), (1, N_points))
    t = rand(Uniform(tmin, tmax), (1, N_points))
    return vcat(x, t) |> gpu_device() .|> Float64
end

"""
the order is t,x,y
"""
function generate_input_t_x_y(config) 
    @unpack N_points, ymin, ymax, xmin, xmax, tmin, tmax = config
    y = rand(Uniform(ymin, ymax), (1, N_points))
    x = rand(Uniform(xmin, xmax), (1, N_points))
    t = rand(Uniform(tmin, tmax), (1, N_points))
    return vcat(t, x, y) |> gpu_device() .|> Float64
end
function generate_input_t_x_y(N_test,config) 
    @unpack N_points, ymin, ymax, xmin, xmax, tmin, tmax = config
    y = rand(Uniform(ymin, ymax), (1, N_test))
    x = rand(Uniform(xmin, xmax), (1, N_test))
    t = rand(Uniform(tmin, tmax), (1, N_test))
    return vcat(t, x, y) |> gpu_device() .|> Float64
end

function generate_input_x_t(N_points, config)
    @unpack xmin, xmax, tmin, tmax = config
    x = rand(Uniform(xmin, xmax), (1, N_points))
    t = rand(Uniform(tmin, tmax), (1, N_points))
    return vcat(x, t) |> gpu_device() .|> Float64
end


function generate_inputboun(config)
    @unpack N_pointsb, xmin, xmax, ymin, ymax = config

    # Generar valores de x solo en los extremos xmin y xmax
    x = reshape(rand([xmin, xmax], N_pointsb), 1, N_pointsb)

    # Generar valores de y aleatorios en el intervalo
    y = reshape(rand(Uniform(ymin, ymax), N_pointsb), 1, N_pointsb)

    # Apilar y mover a GPU
    input = vcat(x, y) |> gpu_device() .|> Float64

    return input
end


function generate_input0_xy(config)
    @unpack N_points0, N_points, xmin, xmax, ymin, ymax, tmin, tmax = config

    x = rand(Uniform(xmin, xmax), (1, N_points0))
    y = rand(Uniform(ymin, ymax), (1, N_points0))
    t = rand(Uniform(tmin, tmax), (1, N_points0))
    t = t*0.0 .+ tmin
    # Stack and move to GPU
    
    input = vcat(t, x, y) |> gpu_device() .|> Float64

    return input
end

# -------------------------------------------------------------------
# Features periódicas y representación con hard enforcement
# -------------------------------------------------------------------
periodic_features(x, L) = vcat(sin.(2π .* x ./ L), cos.(2π .* x ./ L))

"""
u(x,t) = u0(x) + t^2 * Nθ( sin(2πx/L), cos(2πx/L), t )
- Periodicidad en x por construcción (u(x+L,t)=u(x,t)).
- u(x,0) = u0(x) exactamente.
- ∂_t u(x,0) = 0 exactamente (por el factor t^2).
"""
function calculate_periodic_f(x, t, NN, Θ, st)
    u0 = exact_periodic_solution(x, zero(t), config[:A], config[:sigma], config[:x0], config[:c], config[:L])
    ϕx = periodic_features(x, config[:L])             # (2,N)
    nn_in = vcat(ϕx, t)                               # (3,N)
    nn_out = NN(nn_in, Θ, st)[1]                      # (1,N)
    return u0 .+ (t.^2) .* nn_out
end

"""
u(x,t) = u0(x) + t^2 * (L-x)*x* Nθ(x, t )
- Dirichlet en x=0 y x=L por construcción (u(0,t)=u(L,t)=0).
- u(x,0) = u0(x) exactamente.
- ∂_t u(x,0) = 0 exactamente (por el factor t^2).
"""
function calculate_Dirichlet_f(x, t, NN, Θ, st)
    @unpack N_points, xmin, xmax, tmin, tmax, A, B = config
    u0 = A*(x .- xmin).^4 .* (x .- xmax).^4 ./ ((xmax - xmin)/2)^8 # Initial condition
    u1 = -B*(x .- xmin).^3 .* (x .- xmax).^3 ./ ((xmax - xmin)/2)^8 .* (2x .- (xmax - xmin)) # Initial condition for the time derivative
    #u0 = bump.(x, config[:x0], config[:x1], config[:p], config[:A]) # Initial condition
    nn_in = vcat(x, t)
    #@show size(nn_in) size(u0)                              # (3,N)
    nn_out = NN(nn_in, Θ, st)[1]                      # (1,N)
    #f .= ((t - tmin)^2) * (x - xmax) * (x - xmin) #* nn_out
    #@show size(f) size(u0) size(nn_out) 
    return u0 + (t .- tmin) .* u1 + ((t .- tmin).^2) ./ (1.0 .+ (t .- tmin).^2) .* (x .- xmax) .* (x .- xmin) .* nn_out 
end

function calculate_Dirichlet_f_wf(x, t, NN, Θ, st)
    @unpack N_points, xmin, xmax, tmin, tmax, A, B = config
    U0 = A* u0(x) #A*(x .- xmin).^4 .* (x .- xmax).^4 ./ ((xmax - xmin)/2)^8 # Initial condition
    U1 = B* u1(x) #(x .- xmin).^3 .* (x .- xmax).^3 ./ ((xmax - xmin)/2)^8 .* (2x .- (xmax - xmin)) # Initial condition for the time derivative
    #u0 = bump.(x, config[:x0], config[:x1], config[:p], config[:A]) # Initial condition
    nn_in = vcat(x, t)
    #@show size(nn_in) size(u0)                              # (3,N)
    nn_out = NN(nn_in, Θ, st)[1]                      # (1,N)
    #f .= ((t - tmin)^2) * (x - xmax) * (x - xmin) #* nn_out
    #@show size(f) size(u0) size(nn_out) 
    return U0 + (t .- tmin) .* U1 + ((t .- tmin).^2) ./ (1.0 .+ (t .- tmin).^2) .* (x .- xmax) .* (x .- xmin) .* nn_out 
end



"""
    calculate_f(x, y, NN, Θ, st)

Returns the PINN's prediction for u(x, y), enforcing Dirichlet boundary conditions.
"""
function calculate_f(x, y, NN, Θ, st)
    # Hard enforcement of boundary conditions:
    # u(x, 0) = x^2, u(0, y) = y^2, u(1, y) = 1 + y^2, u(x, 1) = 1 + x^2
    return  NN(vcat(x, y), Θ, st)[1]
end
# -------------------------------------------------------------------
# Derivadas por diferencias finitas (segundas en x y t)
# -------------------------------------------------------------------
function calculate_derivatives_Dirichlet(x, t, NN, Θ, st)
    #@unpack N_points_x. N_points_t, xmin, xmax, tmin, tmax = config
    ϵ = ∜(eps())  # paso óptimo para 2ª derivada aprox.

    f      = calculate_Dirichlet_f_wf(x, t, NN, Θ, st)
    fxp    = calculate_Dirichlet_f_wf(x .+ ϵ, t, NN, Θ, st)
    fxm    = calculate_Dirichlet_f_wf(x .- ϵ, t, NN, Θ, st)
    ftp    = calculate_Dirichlet_f_wf(x, t .+ ϵ, NN, Θ, st)
    ftm    = calculate_Dirichlet_f_wf(x, t .- ϵ, NN, Θ, st)

    ∂2f_∂x2 = (fxp .- 2 .* f .+ fxm) / ϵ^2
    ∂2f_∂t2 = (ftp .- 2 .* f .+ ftm) / ϵ^2
    return f, ∂2f_∂x2, ∂2f_∂t2
end


function calculate_fields_and_derivatives_Toy_MHD(t, x, y, NN, Θ, st)
    ϵ = ∜(eps())  # paso óptimo para 1ª derivada aprox.

    B1      = NN(vcat(t, x, y), Θ, st)[1][1,:]
    B2      = NN(vcat(t, x, y), Θ, st)[1][2,:]

    DtB1    = (NN(vcat(t .+ ϵ, x, y), Θ, st)[1][1,:] .- NN(vcat(t .- ϵ, x, y), Θ, st)[1][1,:]) / (2ϵ)
    DtB2    = (NN(vcat(t .+ ϵ, x, y), Θ, st)[1][2,:] .- NN(vcat(t .- ϵ, x, y), Θ, st)[1][2,:]) / (2ϵ)
    DxB1    = (NN(vcat(t, x .+ ϵ, y), Θ, st)[1][1,:] .- NN(vcat(t, x .- ϵ, y), Θ, st)[1][1,:]) / (2ϵ)
    DxB2    = (NN(vcat(t, x .+ ϵ, y), Θ, st)[1][2,:] .- NN(vcat(t, x .- ϵ, y), Θ, st)[1][2,:]) / (2ϵ)
    DyB1    = (NN(vcat(t, x, y .+ ϵ), Θ, st)[1][1,:] .- NN(vcat(t, x, y .- ϵ), Θ, st)[1][1,:]) / (2ϵ)
    DyB2    = (NN(vcat(t, x, y .+ ϵ), Θ, st)[1][2,:] .- NN(vcat(t, x, y .- ϵ), Θ, st)[1][2,:]) / (2ϵ)

    return B1, B2, DtB1, DtB2, DxB1, DxB2, DyB1, DyB2
end

function calculate_V_and_DVs(t,x,y)
    #@unpack N_points, xmin, xmax, ymin, ymax = config
    r0 = 0.16
    x = vec(x)
    y = vec(y)
    r2(x,y) = (x-0.5)^2 + (y-0.5)^2
    V1 = (x.-0.5).*r2.(x,y).*(r2.(x,y) .- r0)
    V2 = (y.-0.5).*r2.(x,y).*(r2.(x,y) .- r0)
    DxV1 = r2.(x,y).*(r2.(x,y) .- r0) .+ 2 .*(x.-0.5).^2 .*(2 .*r2.(x,y) .- r0)
    DxV2 = (y.-0.5).*2 .*(x.-0.5).*(2 .*r2.(x,y) .- r0)
    DyV1 = (x.-0.5).*2 .*(y.-0.5).*(2 .*r2.(x,y) .- r0)
    DyV2 = r2.(x,y).*(r2.(x,y) .- r0) .+ 2 .*(y.-0.5).^2 .*(2 .*r2.(x,y) .- r0)
    return V1, V2, DxV1, DxV2, DyV1, DyV2
end

function bump(x,x0,x1,p,A)
    if x > x0 && x < x1
    return A*(x-x0)^p * (x1-x)^p / ((x1-x0)/2)^(2p)
    #return A*NaNMath.pow(x-x0,p)*NaNMath.pow(x-x1,p)/NaNMath.pow((x1-x0)/2,2p)
    else
        return 0.0
    end
end


function bump_x(x,x0,x1,p,A) #x derivative of b
    if x > x0 && x < x1
    return p*A*((x1-x0)/2)^(-2p)*(x-x0)^(p-1)*(x-x1)^(p-1)*(2x-x0-x1)
    else
        return 0.0
    end
end

"""
adaptive_rad:
  - Ntest: nº de candidatos uniformes que se generan
  - Nint: nº de puntos que quieres seleccionar para entrenar
  - k1, k2: hiperparámetros RAD (ponderación por |residuo|^k1, desplazamiento k2)
Devuelve un 'input' de tamaño (2, Nint) ponderado por el residuo.
"""
function adaptive_rad(NN, Θ, st, config; Ntest=50_000, Nint=config[:N_points], k1=1.0, k2=1e-6)
    Xtest = generate_input_x_t(Ntest, config)
    Y = residual_at_points_Dirichlet(Xtest, NN, Θ, st)        # |residuo| en cada punto
    w = (Y .^ k1)
    w = w ./ mean(w) .+ k2                                  # normalización + desplazamiento
    p = w ./ sum(w)                                         # distribución de probabilidad
    ids = sample(1:length(p), Weights(p), Nint; replace=false)
    return Xtest[:, ids]                                     # (2, Nint)
end

"""
adaptive_rad_toy_MHD:
  - Ntest: nº de candidatos uniformes que se generan
  - Nint: nº de puntos que quieres seleccionar para entrenar
  - k1, k2: hiperparámetros RAD (ponderación por |residuo|^k1, desplazamiento k2)
Devuelve un 'input' de tamaño (2, Nint) ponderado por el residuo.
"""
function adaptive_rad_toy_MHD(NN, Θ, st, config; Ntest=50_000, Nint=config[:N_points], k1=1.0, k2=1e-6)
    Xtest = generate_input_t_x_y(Ntest, config)
    Y, _ = residual_at_points_Toy_MHD(Xtest, NN, Θ, st)        # |residuo| en cada punto
    YR = reshape(Y, Ntest, 3)
    w = ((abs.(YR[:,1]) + abs.(YR[:,2]) + abs.(YR[:,3])) .^ k1)
    w = w ./ mean(w) .+ k2                                  # normalización + desplazamiento
    p = w ./ sum(w)                                         # distribución de probabilidad
    ids = sample(1:length(p), Weights(p), Nint; replace=false)
    return Xtest[:, ids]                                     # (2, Nint)
end