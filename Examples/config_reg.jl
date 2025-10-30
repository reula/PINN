# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------
config_basic = Dict(
    :N_input => 1,          # [x; t]
    :N_neurons => 20,
    :N_layers => 1,
    :N_output => 1, 
    :N_points => 500,     # puntos de colisión (x)
    :xmin => 0.0,
    :xmax => 1.0,           # = L dominio espacial
    #:optimizer => BFGS(),
    #:optimizer => SSBroyden(Optim.Options(linesearch=LineSearches.HagerZhang(), show_trace=true)),
    :optimizer => SSBroyden(),
    #:optimizer => AdaMax(; alpha=0.001, beta_mean=0.9, beta_var=0.999, epsilon=1e-8),
    #:optimizer => Adam(; alpha=0.002, beta_mean=0.9, beta_var=0.999, epsilon=1e-8),
    :maxiters => 3_000,
    :N_rounds => 10,    # 5           # nº de rondas RAD
    :iters_per_round => 500,       # iteraciones BFGS por ronda
    :k1 => 1.0, 
    :k2 => 1.0,        # hiperparámetros RAD
    :N_test => 4_000,             # candidatos por ronda, mayor que N_points
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


