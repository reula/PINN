# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------
config_basic = Dict(
    :N_input => 2,          # [x; t]
    :N_neurons => 20,
    :N_layers => 3,
    :N_output => 1, 
    :N_points => 2_000,     # puntos de colisión (x,t)
    :N_points_bound => 0, # puntos de frontera
    :N_points_0 => 0,    # puntos de condición inicial
    :xmin => 0.0,
    :xmax => 1.0,           # = L dominio espacial
    :tmin => 0.0,           # t_min
    :tmax => 2.0,           # t_max
    #:optimizer => BFGS(),
    #:optimizer => SSBroyden(Optim.Options(linesearch=LineSearches.HagerZhang(), show_trace=true)),
    #:optimizer => SSBroyden(),
    :optimizer => AdaMax(; alpha=0.001, beta_mean=0.9, beta_var=0.999, epsilon=1e-8),
    #:optimizer => Adam(; alpha=0.002, beta_mean=0.9, beta_var=0.999, epsilon=1e-8),
    :maxiters => 3_000,
    :N_rounds => 40,    # 5           # nº de rondas RAD
    :iters_per_round => 1000,       # iteraciones BFGS por ronda
    :k1 => 1.0, 
    :k2 => 1.0,        # hiperparámetros RAD
    :N_test => 8_000,             # candidatos por ronda, mayor que N_points
    #:method => :adaptive,
    :method => :direct,
    # for the initial data
    :A => 1.0,
    :B => 0.0,
    :x0 => 2.0,
    :x1 => 3.0,
    :p => 8,
    :c => 1.0
)


config_test = Dict( # a minimal configuration to test the code
    :N_input => 2,          # [x; t]
    :N_neurons => 20,
    :N_layers => 4,
    :N_output => 1, 
    :N_points => 2_000,     # puntos de colisión (x,t)
    :N_points_bound => 0, # puntos de frontera
    :N_points_0 => 0,    # puntos de condición inicial
    :xmin => 0.0,
    :xmax => 1.0,           # = L dominio espacial
    :tmin => 0.0,           # t_min
    :tmax => 2.0,           # t_max
    :optimizer => BFGS(),
    #:optimizer => SSBroyden(),
    :maxiters => 3_000,
    :N_rounds => 50,    # 5           # nº de rondas RAD
    :iters_per_round => 10,       # iteraciones BFGS por ronda
    :k1 => 1.0, 
    :k2 => 1.0,        # hiperparámetros RAD
    :N_test => 3_000,             # candidatos por ronda, mayor que N_points
    :method => :adaptive,
    #:method => :direct,
    # for the initial data
    :A => 1.0,
    :B => 0.0,
    :x0 => 2.0,
    :x1 => 3.0,
    :p => 8,
    :c => 1.0
)
