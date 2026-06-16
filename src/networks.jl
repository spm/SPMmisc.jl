
function RUnet2(c::Vector; cout=c[1])
    nonlin = identity
    WR     =          Conv((4,4), c[1]=>c[2],  bias=true,  stride=2, pad=(1,1,1,1))
    WP     = ConvTranspose((4,4), c[2]=>cout,  bias=true,  stride=2, pad=(1,1,1,1))
    idinit(n) = randn(Float32, 3, 3, n, n)*sqrt(1f0/(3*3*n))/2;
    M()    = SkipConnection(Conv(idinit(c[2]), true, pad=1), (a,b)-> max.(a,b))
    nrm    = InstanceNorm(c[2])

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


"""
    RUnet3(c::Vector; cout=c[1])

3D residual U-net, using 2x2x2 filters for restriction and prolongation.
Note that image dimensions must be even, except for the deepest layer.
"""
function RUnet3(c::Vector; cout=c[1])
    WR     =          Conv((2,2,2), c[1]=>c[2],  bias=true,  stride=2)
    WP     = ConvTranspose((2,2,2), c[2]=>cout,  bias=true,  stride=2)

    idinit(n) = (t=randn(Float32, 3, 3, 3, n, n)*sqrt(1f0/(3^3*n))/2;
                 t[2,2,2,:,:] .= t[2,2,2,:,:] + I; t)
    M()  = SkipConnection(Conv(idinit(c[2]), true, pad=1), (a,b)-> max.(a,b))

    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(), M(), SkipConnection(RUnet3(c[2:end]), +),
                    M(), M(), WP)
    else
        net = Chain(WR, nrm, M(), M(), M(), M(), WP)
    end
    return net
end


"""
    pRUnet3(c::Vector; cout=c[1])

3D residual U-net, but with an intensity normalisation of the input.
"""
function pRUnet3(c::Vector; cout=c[1])
    net = Chain(scalnorm, RUnet3(c, cout=cout)...)
end

"""
    BUnet3(c::Vector; cout=c[1])

Bigger 3D residual U-net, using 4x4x4 filters for restriction and prolongation.
Note that image dimensions must be even, except for the deepest layer.
"""
function BUnet3(c::Vector; cout=c[1])
    WR     =          Conv((4,4,4), c[1]=>c[2],  bias=true,  stride=2, pad=1)
    WP     = ConvTranspose((4,4,4), c[2]=>cout,  bias=true,  stride=2, pad=1)

    idinit(n) = (t=randn(Float32, 3, 3, 3, n, n)*sqrt(1f0/(3^3*n))/2;
                 t[2,2,2,:,:] .= t[2,2,2,:,:] + I; t)
    M()  = SkipConnection(Conv(idinit(c[2]), true, pad=1), (a,b)-> max.(a,b))

    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(), M(), SkipConnection(BUnet3(c[2:end]), +),
                    M(), M(), WP)
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


export RUnet2, pRUnet2, RUnet3, pRUnet3, BUnet3, pBUnet3, scalnorm


