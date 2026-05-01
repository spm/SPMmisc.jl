using Flux, JLD2, NIfTI

function MorphoUnet(c::Vector; cout=c[1])
    nonlin = identity
    WR     =          Conv((4,4), c[1]=>c[2],  bias=true,  stride=2, pad=(1,1,1,1))
    WP     = ConvTranspose((4,4), c[2]=>cout,  bias=true,  stride=2, pad=(1,1,1,1))
   #idinit(n) = (t=randn(Float32, 3, 3, n, n)*sqrt(1f0/(3*3*n))/2; t[2,2,:,:] .= t[2,2,:,:] + I; t)
    idinit(n) = randn(Float32, 3, 3, n, n)*sqrt(1f0/(3*3*n))/2
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

function loadnet(filename)
    print("Loading ", filename)
    sett = JLD2.load(filename, "settings")
    net  = sett[3](sett[1]...; sett[2]...)
    W1   = JLD2.load(filename, "W")
    W0   = Flux.trainables(net)
    for i=1:length(W0)
        W0[i] .= W1[i]
    end
    print("\n")
    return net, sett
end


function nan2zero!(x)
    x[isnan.(x)] .= 0
    x
end


function segment(P,net,sett)
    print(basename(P)," ")
    nii = niread(P)
    X   = Float32.(nii.raw[:,:,:,1])
    mat = NIfTI.getaffine(nii)
    vx2 = sqrt.(sum(mat[1:3,1:3].^2,dims=1))
    d2  = size(X)

    nc  = sett[2].cout
    Y   = similar(X,(d2[1],d2[2],d2[3],nc))
    for k=1:d2[3]
        x          = nan2zero!(reshape(X[:,:,k],(d2[1],d2[2],1,1)))
        Y[:,:,k,:] = net(x)
        if rem(k,10)==0; print("."); end
    end
    Y .= Flux.softmax(Y, dims=4)

    hdr           = deepcopy(nii.header)
    hdr.descrip   = NIfTI.string_tuple("2D NN segmented", 80)
    hdr.datatype  = 2
    hdr.scl_slope = 1/255
    hdr.scl_inter = 0
    hdr.bitpix    = 8

    dim           = [hdr.dim...]
    dim[1]        = ndims(Y)
    dim[5]        = nc
    dim[6:end]   .=1
    hdr.dim       = (dim...,)

    Y             = UInt8.(round.(Y./hdr.scl_slope))
    nio           = NIVolume(hdr,Y[:,:,:,1:nc])
    fname         = joinpath(dirname(P), "c00" * basename(P))
    niwrite(fname,nio)
    print("\n"); flush(stdout)
    return fname
end

function apply2Dnet(network::String,images::Array{String})
    net,sett = loadnet(network)
    for f in images
        segment(f,net,sett)
    end
end


