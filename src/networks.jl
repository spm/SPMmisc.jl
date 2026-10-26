
function RUnet2(c::Vector; cout=c[1])
    nonlin = identity
    WR  =          Conv((4,4), c[1]=>c[2],  bias=false, stride=2, pad=(1,1,1,1))
    WP  = ConvTranspose((4,4), c[2]=>cout,  bias=true,  stride=2, pad=(1,1,1,1))
    M() = SkipConnection(Chain(relu, Conv((3,3), c[2]=>c[2], bias=true, pad=1)), +)
    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(), M(),
                    SkipConnection(RUnet2(c[2:end]), +), M(), M(), WP)
    else
        net = Chain(WR, nrm, M(), M(), M(), M(), WP)
    end
    return net
end
function pRUnet2(c::Vector; cout=c[1])
    net = Chain(scalnorm, RUnet2(c, cout=cout)...)
end


function RUnet3old(c::Vector; cout=c[1])
    WR  =          Conv((2,2,2), c[1]=>c[2],  bias=false,  stride=2)
    WP  = ConvTranspose((2,2,2), c[2]=>cout,  bias=true,  stride=2)
    M() = SkipConnection(Chain(relu, Conv((3,3,3), c[2]=>c[2], bias=true, pad=1)), +)
    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(), M(), SkipConnection(RUnet3old(c[2:end]), +),
                    M(), M(), WP)
    else
        net = Chain(WR, nrm, M(), M(), M(), M(), WP)
    end
    return net
end
function pRUnet3old(c::Vector; cout=c[1])
    net = Chain(scalnorm, RUnet3old(c, cout=cout)...)
end


"""
    RUnet3(c::Vector; cout=c[1])

3D residual U-net, using 2x2x2 filters for restriction and prolongation.
Note that image dimensions must be even, except for the deepest layer.
"""
function RUnet3(c::Vector; cout=c[1],  rp=2)
    if rp==2
        WR  =          Conv((2,2,2), c[1]=>c[2],  bias=true, stride=2)
        WP  = ConvTranspose((2,2,2), c[2]=>cout,  bias=true, stride=2)
    else
        WR  =          Conv((4,4,4), c[1]=>c[2],  bias=true, stride=2, pad=1)
        WP  = ConvTranspose((4,4,4), c[2]=>cout,  bias=true, stride=2, pad=1)
    end
    M() = SkipConnection(Chain(relu, Conv((3,3,3), c[2]=>c[2], bias=true, pad=1)), +)
    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(), M(), SkipConnection(RUnet3(c[2:end]; rp=rp), +), M(), M(), WP)
    else
        net = Chain(WR, nrm, M(), M(), M(), M(), WP)
    end
    return net
end

"""
    pRUnet3(c::Vector; cout=c[1])

3D residual U-net, but with an intensity normalisation of the input.
"""
function pRUnet3(c::Vector; cout=c[1], rp=2)
    net = Chain(scalnorm, RUnet3(c, cout=cout, rp=rp)...)
end

function RUnnnet3(c::Vector; cout=c[1],  rp=2)
    if rp==2
        WR  =          Conv((2,2,2), c[1]=>c[2],  bias=true, stride=2)
        WP  = ConvTranspose((2,2,2), c[2]=>cout,  bias=true, stride=2)
    else
        WR  =          Conv((4,4,4), c[1]=>c[2],  bias=true, stride=2, pad=1)
        WP  = ConvTranspose((4,4,4), c[2]=>cout,  bias=true, stride=2, pad=1)
    end
    M() = SkipConnection(Chain(relu, Conv((3,3,3), c[2]=>c[2], bias=true, pad=1)), +)

    if length(c)>2
        net = Chain(WR, M(), M(), SkipConnection(RUnnnet3(c[2:end]; rp=rp), +), M(), M(), WP)
    else
        net = Chain(WR, M(), M(), M(), M(), WP)
    end
    return net
end

"""
    BUnet3(c::Vector; cout=c[1])

Bigger 3D residual U-net, using 4x4x4 filters for restriction and prolongation.
Note that image dimensions must be even, except for the deepest layer.
"""
function BUnet3(c::Vector; cout=c[1])
    WR  =          Conv((4,4,4), c[1]=>c[2],  bias=true, stride=2, pad=1)
    WP  = ConvTranspose((4,4,4), c[2]=>cout,  bias=true, stride=2, pad=1)
    M() = SkipConnection(Chain(relu, Conv((3,3,3), c[2]=>c[2], bias=true, pad=1)), +)
    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(), M(), SkipConnection(BUnet3(c[2:end]), +), M(), M(), WP)
    else
        net = Chain(WR, nrm, M(), M(), M(), M(), WP)
    end
    return net
end

"""
    pBUnet3(c::Vector; cout=c[1])

Bigger 3D residual U-net, but with an intensity normalisation of the input.
"""
function pBUnet3(c::Vector; cout=c[1])
    net = Chain(scalnorm, BUnet3(c, cout=cout)...)
end


"""
    scalnorm(x)

Intensity normalisation of x.
"""
function scalnorm(x)
    dims = 1:(ndims(x)-2)
    s    = 1 ./ sqrt.((sum(x.^2, dims=dims) .+ 1) / prod(size(x)[dims]))
    return(x .* s)
end

"""
    loadnet(filename)

Load a neural network from a file.
Network architectures are described by some function that must
be avaiable at the time of loading.
"""
function loadnet(filename)
    if isempty(dirname(filename))
        filename = joinpath(@__DIR__, "..", "artifacts", filename)
    end

    print("Loading ", basename(filename))

    sett = JLD2.load(filename, "settings")
    #sett = (sett[1:2]..., pBUnet3) # Temporary fix
    net  = sett[3](sett[1]...; sett[2]...)
    W1   = JLD2.load(filename, "W")
    W0   = Flux.trainables(net)
    for i=1:length(W0)
        W0[i] .= W1[i]
    end
    print("\n")
    return net, sett
end

function savenet(net, filename, sett)
    W = Flux.trainables(net)
    JLD2.jldsave(filename; W = W, settings=sett)
end

export loadnet, RUnet2, pRUnet2, RUnet3, pRUnet3, BUnet3, pBUnet3, RUnnnet3, scalnorm


