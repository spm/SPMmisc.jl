function parts(P)
    ind = [1,1,1,1]
    mch = match(r"^(?<name>[^,]*)$", P)
    if mch == nothing
        mch   = match(r"^(?<name>[^,]*)" * "," * r"(?<indices>[\d,]*)$", P)
        if mch == nothing
            error("Bollox")
        end
        fname   = mch[:name]
        indices = mch[:indices]
        for i=1:4
            mch   = match(r"^(?<num>[^,]*)" * "," * r"(?<indices>[\d,]*)$", indices)
            if mch == nothing
                ind[i] = parse(Int64, indices)
                break
            end
            ind[i] = parse(Int64, mch[:num])
            indices = mch[:indices]
        end
    else
        fname = mch[:name]
    end
    return fname, ind
end


function NNseg3D(P, net, sett, fname)
    print(fname," ")

    nam,ind = parts(P[1])
    nii = niread(nam)
    Xi  = Float32.(nii.raw[:,:,:,ind...]).*nii.header.scl_slope .+ nii.header.scl_inter
    Xi  = reshape(Xi,(size(Xi)[1:3]...,1,1))
    X    = Xi
    mat  = NIfTI.getaffine(nii)
    vx2  = sqrt.(sum(mat[1:3,1:3].^2,dims=1))
    d2   = size(X)[1:3]
    hdr  = nii.header
    fnam = nam

    print("."); flush(stdout)

    for i=2:length(P)
        nam,ind = parts(P[i])
        nii = niread(nam)
        Xi  = Float32.(nii.raw[:,:,:,ind...]).*nii.header.scl_slope .+ nii.header.scl_inter
        Xi  = reshape(Xi,(size(Xi)[1:3]...,1,1))
        X   = cat(X, Xi, dims=4)
        print("."); flush(stdout)
    end

    # Restrictions need even-sized voxel dimensions, so see how many
    # are done and make the image dimensions a multiple of 2^n.
    nc       = sett[2].cout
    divby    = 2^(length(sett[1][1])-1)

    dn  = ceil.(d2[1:3].*vx2[:])
    dn  = (Int64.(ceil.(dn[1:3]./divby).*divby)...,)
    Aff = Float32.([vx2[1] 0 0 0; 0 vx2[2] 0 0; 0 0 vx2[3] 0; 0 0 0 1])
    x   = affine_pull(X, inv(Aff), dn, Settings(1,1,1))
    print("o"); flush(stdout)
    y   = net(x)
    print("."); flush(stdout)
    y   = affine_pull(y,     Aff,  d2, Settings(1,1,1))
    if size(y)[4] > 1
        Y   = Flux.softmax(y, dims=4)
        Y   = Y[:,:,:,1:(end-1),:]
    else
        Y   = Flux.sigmoid(y)
    end

    hdr.descrip   = NIfTI.string_tuple("NNseg3D", 80)

    # Write as UInt8
    hdr.datatype  = 2
    hdr.scl_slope = 1/255
    hdr.scl_inter = 0
    hdr.bitpix    = 8
    Y             = UInt8.(round.(Y./hdr.scl_slope))
    dim           = [hdr.dim...]
    dim[1]        = ndims(Y)
    dim[5]        = size(Y,4)
    dim[6:end]   .=1
    hdr.dim       = (dim...,)

    nio           = NIVolume(hdr,Y[:,:,:,1:end])
    niwrite(fname, nio)
    print("\n"); flush(stdout)
    return fname
end


"""
    do_depth(images::Array{Array{String}}, network::String="CorticalDepth.jld2")

Apply a 3D segmentation network to GM and WM images. Results
are saved to a NIfTI file called depth_*.nii.
"""
function do_depth(images, network::String="CorticalDepth.jld2")
    net,sett = loadnet(network)
    for f in images
        if isa(f,String)
            f = [f,]
        end
        fnam,ind = parts(f[1])
        fname    = joinpath(dirname(fnam), "depth_" * basename(fnam)[3:end])
        NNseg3D(f, net, sett, fname)
    end
end


"""
    apply3net(images::Array{String}, network::String)

    apply3net(images::Array{Array{String}}, network::String="GWC-any.jld2")

Apply a 3D segmentation network to images. Results
are saved to a 4D NIfTI file called c00*.nii.
"""
function apply3Dnet(images, network::String="GWC-any.jld2")
    net,sett = loadnet(network)
    for f in images
        if isa(f,String)
            f = [f,]
        end
        fnam,ind = parts(f[1])
        fname    = joinpath(dirname(fnam), "c00" * basename(fnam))
        NNseg3D(f, net, sett, fname)
    end
end

