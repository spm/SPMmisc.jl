module SPMmisc
using Flux, JLD2, NIfTI, LinearAlgebra

include("networks.jl")
include("apply2Dnet.jl")
include("apply3Dnet.jl")
include("cort.jl")

export apply2Dnet, apply3Dnet
end

