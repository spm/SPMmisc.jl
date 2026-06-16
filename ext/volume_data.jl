using Random
using NIfTI
include("random_patch.jl")

function header(P)
    io = open(P,"r")
    hdr, swapped = NIfTI.read_header(io)
    close(io)
    hdr
end

mutable struct Vol
    volume::Array{Float32,3}
    mat::Matrix{Float32}
end

mutable struct VolData
    Px::Tuple
    Py::Tuple
    volX::Vector{Vol}
    volY::Vector{Vol}
    patchsize::Tuple
    batchsize::Integer
    batchcount::Integer
    nviews::Integer
    current::Integer
    order::Vector
    augment::Function
end

import LinearAlgebra.I
function eye(N=4)
    Float32.(I(N))
end

function VolData(Px,Py; patchsize::Tuple=(96,96,96), batchsize::Integer=1, augment = augment_seg)
    empty = [Vol(zeros(0,0,0),eye(4))]
    order = randperm(length(Px[1]))
    VolData(Px,Py,empty,empty,patchsize,batchsize,0,2,0,order,augment)
end


function load_volumes!(Px::Tuple, num=1)
    Nc   = length(Px)
    vols = Vector{Vol}(undef,32)
    c1   = 0
    for c=1:Nc
        nii    = niread(Px[c][num])
        for ii=1:size(nii.raw,4)
            volume = Float32.(nii.raw[:,:,:,ii]).*nii.header.scl_slope .+
                              nii.header.scl_inter
            mat    = NIfTI.getaffine(nii) # Need to patch this for 1-offsets
            c1    += 1
            vols[c1] = Vol(volume, mat)
            flush(stdout)
        end
    end
    vols = vols[1:c1]
    flush(stdout)
    return vols
end


function next!(data::VolData)
    data.batchcount = rem(data.batchcount, data.nviews) + 1
    if data.batchcount==1
        data.current += 1
        if data.current>length(data.Px[1])
            data.current = 0
            data.order .= randperm(length(data.Px[1]))
            return nothing, nothing
        end
        data.volX = load_volumes!(data.Px, data.order[data.current])
        data.volY = load_volumes!(data.Py, data.order[data.current])
    end
    mat = [1f0 0f0 0f0 0f0
           0f0 1f0 0f0 0f0
           0f0 0f0 1f0 0f0
           0f0 0f0 0f0 1f0] # Unused
    x0,y0 = random_patch(data.volX, data.volY,(data.patchsize...,1),mat; zoom_sd=0.2)
    x,y   = data.augment(x0,y0)
    return x,y
end


import Base.show
function Base.show(io::IO, sd::VolData)
    print(io,"Curr: ", sd.current, "; batchsize=", sd.batchsize, "\n")
    return nothing
end

