
function MorphoUnet(c::Vector; cout=c[1])
    nonlin = identity
    WR     =          Conv((4,4), c[1]=>c[2],  bias=true,  stride=2, pad=(1,1,1,1))
    WP     = ConvTranspose((4,4), c[2]=>cout,  bias=true,  stride=2, pad=(1,1,1,1))
   #idinit(n) = (t=randn(Float32, 3, 3, n, n)*sqrt(1f0/(3*3*n))/2; t[2,2,:,:] .= t[2,2,:,:] + I; t)
    idinit(n) = randn(Float32, 3, 3, n, n)*sqrt(1f0/(3*3*n))/2;
    M(n)   = SkipConnection(Conv(idinit(n), true, pad=1), (a,b)-> max.(a,b))

    nrm = InstanceNorm(c[2])

    if length(c)>2
        net = Chain(WR, nrm, M(c[2]), M(c[2]), M(c[2]), M(c[2]),
                    SkipConnection(MorphoUnet(c[2:end]), +), WP)
    else
        net = Chain(WR, nrm, M(c[2]), M(c[2]), M(c[2]), M(c[2]), WP)
    end
    return net
end

function scalnorm(x)
    s = 1 ./ sqrt.(sum(x.^2, dims=(1,2)) / prod(size(x)[1:2]) .+ 1)
    return(x .* s)
end

function pMorphoUnet(c::Vector; cout=c[1])
    net = Chain(scalnorm, MorphoUnet(c, cout=cout)...)
end

function lse(x)
    dim = ndims(x)-1
    mx  = maximum(x,dims=dim)
    l   = log.(sum(exp.(x .- mx),dims=dim)) .+ mx
end

function gmmadjust(q, x)
    @assert(size(x)[ndims(x)-1] == 1)
    dims = 1:(ndims(x)-2)
    p    = exp.(q .- lse(q))
    s0   = sum(p,    dims=dims) .+ 1f-3
    s1   = sum(p.*x, dims=dims)
    m    = s1./s0
    r2   = (x.-m).^2
    v    = (sum(p.*r2, dims=dims) .+ 1f-3)./s0
    ll   = .-r2 ./ (2v) .- log.(v)./2 .+ q
    ll   = ll .- lse(ll)
end

function experimental0(nf; cout=nf[1], cmid=8)
    nf2    = deepcopy(nf)
    nf2[1] = cmid
    unet1  = pMorphoUnet(nf ; cout=cmid)
    unet2  =  MorphoUnet(nf2; cout=cout)
    net    = Chain(unet1, unet2)
end

function experimental1(nf; cout=nf[1], cmid=8)
    nf2    = deepcopy(nf)
    nf2[1] = cmid
    unet1  = pMorphoUnet(nf ; cout=cmid)
    unet2  =  MorphoUnet(nf2; cout=cout)
    net    = Chain(SkipConnection(unet1,gmmadjust), unet2)
end

function experimental2(nf; cout=nf[1], cmid=8)
    nf2    = deepcopy(nf)
    nf2[1] = cmid
    unet1  = pMorphoUnet(nf ; cout=cmid)
    unet2  =  MorphoUnet(nf2; cout=cout)
    net    = SkipConnection(Chain(SkipConnection(unet1, gmmadjust), unet2), gmmadjust)
end

twopass  = experimental1
ablation = experimental0

export MorphoUnet, pMorphoUnet, scalnorm
export experimental0, experimental1, experimental2


