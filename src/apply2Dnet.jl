using Flux, JLD2, NIfTI
include("networks.jl")

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


