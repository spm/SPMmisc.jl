module SPMmisc
using Flux, JLD2, NIfTI, LinearAlgebra
using PushPull

include("networks.jl")
include("apply2Dnet.jl")
include("apply3Dnet.jl")
include("cort.jl")

export apply2Dnet, apply3Dnet, do_depth
end

