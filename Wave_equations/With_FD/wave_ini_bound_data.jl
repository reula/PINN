# here we write the initial and boundary data for hard imposition so that we can also use from the post_process 
using Revise
using UnPack
includet("config.jl")

@unpack xmin, xmax, A, B, p = config

u0(x) = A*sin.(2π*x*p/(xmax - xmin))
u1(x) = 0.0.*x

# boundary data heres