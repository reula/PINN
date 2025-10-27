# here we write the initial and boundary data for hard imposition so that we can also use from the post_process 
using Revise
using UnPack
includet("config.jl")

@unpack xmin, xmax, A, B, p = config

u0(x) = A*sin.(2π*x*p / (xmax - xmin))
u1(x) = B*sin.(2π*x*p / (xmax - xmin))


#u0(x) = A*(x .- xmin).^p .* (x .- xmax).^p ./ ((xmax - xmin)/2)^(2p) # Initial condition
#u1(x) = -B*p*(x .- xmin).^(p-1) .* (x .- xmax).^(p-1) ./ ((xmax - xmin)/2)^(2p) .* (2x .- (xmax - xmin)) # Initial condition for the time derivative

# boundary data heres