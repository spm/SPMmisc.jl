function segment3D(P,net,sett)
    print(basename(P)," ")
    nii = niread(P)
    X   = Float32.(nii.raw[:,:,:,1]).*nii.header.scl_slope .+ nii.header.scl_inter
    X   = reshape(X,(size(X)[1:3]...,size(X,4),1)) #.* (length(X)/sum(X))
    #X[~isfinite(X)] .= 0
    mat = NIfTI.getaffine(nii)
    vx2 = sqrt.(sum(mat[1:3,1:3].^2,dims=1))[:]
    d2  = size(X)[1:3]

    # Restrictions need even-sized voxel dimensions, so see how many
    # are done and make the image dimensions a multiple of 2^n.
    nc       = sett[2].cout
    divby    = 2^(length(sett[1][1])-1)

    if false
        # Work with original voxel sizes
        function reflect(i, d)
            j = mod(i-1, 2d)
            b = UInt64((j<d) ? j+1 : 2d-j)
        end

        dn  = Int64.(ceil.(d2[1:3]./divby).*divby)
        t   = Int64.(ceil.((dn[1:3] .- d2[1:3])./2))
        x   = similar(X,(dn...,size(X,4),1))
        x[1:dn[1],1:dn[2],1:dn[3],:,1] = X[reflect.((1:dn[1]).-t[1],d2[1]),
                                           reflect.((1:dn[2]).-t[2],d2[2]),
                                           reflect.((1:dn[3]).-t[3],d2[3]),:,1]
        y   = net(x)
        Y   = Flux.softmax(y[(1:d2[1]).+t[1], (1:d2[2]).+t[2], (1:d2[3]).+t[3],:,1], dims=4)
    else
        # Work with 1 mm isotropic voxels
        # todo: generalise to arbitrary voxel sizes (and orientations)

        dn  = ceil.(d2[1:3].*vx2)
        dn  = (Int64.(ceil.(dn[1:3]./divby).*divby)...,)
        Aff = Float32.([vx2[1] 0 0 0; 0 vx2[2] 0 0; 0 0 vx2[3] 0; 0 0 0 1])
        x   = affine_pull(X, inv(Aff), dn, Settings(1,1,1))
        y   = net(x)
        y   = affine_pull(y,     Aff,  d2, Settings(1,1,1))
        Y   = Flux.softmax(y, dims=4)
    end

    hdr           = deepcopy(nii.header)
    hdr.descrip   = NIfTI.string_tuple("NN segmented", 80)
    if true
        # Write as UInt8
        hdr.datatype  = 2
        hdr.scl_slope = 1/255
        hdr.scl_inter = 0
        hdr.bitpix    = 8
        Y             = UInt8.(round.(Y./hdr.scl_slope))
    else
        # Write as Float32
        hdr.datatype  = 16
        hdr.scl_slope = 1
        hdr.scl_inter = 0
        hdr.bitpix    = 32
    end
    dim           = [hdr.dim...]
    dim[1]        = ndims(Y)
    dim[5]        = size(Y,4)-1
    dim[6:end]   .=1
    hdr.dim       = (dim...,)

    nio           = NIVolume(hdr,Y[:,:,:,1:(end-1)])
    fname         = joinpath(dirname(P), "c00" * basename(P))
    niwrite(fname,nio)
    print("\n"); flush(stdout)
    return fname
end

"""
    apply3net(network::String,images::Array{String})

Apply a 3D segmentation network to images. Results
are saved to a 4D NIfTI file called c00*.nii.
"""
function apply3Dnet(network::String,images::Array{String})
    net,sett = loadnet(network)
    for f in images
        segment3D(f,net,sett)
    end
end

