using Flux, JLD2, NIfTI
using SPMmisc
include("networks.jl")
include("loadnet.jl")


function segment3D(P,net,sett)
    print(basename(P)," ")
    nii = niread(P)
    X   = Float32.(nii.raw[:,:,:,1])
    X   = reshape(X,(size(X)...,1)) #.* (length(X)/sum(X))
    #X[~isfinite(X)] .= 0
    mat = NIfTI.getaffine(nii)
    vx2 = sqrt.(sum(mat[1:3,1:3].^2,dims=1))
    d2  = size(X)

    nc       = sett[2].cout
    divby    = 2^(length(sett[1][1])-1)

    if false
        function starts(di,dp)
            ((1:ceil(2*di/dp)).-1).*dp/2 .- dp/2
        end

        Y   = similar(X,(d2[1],d2[2],d2[3],nc,1))
        W   = zeros(Float32,d2[1],d2[2],d2[3])
        dn = Int64.(floor.(d2[1:3]./divby).*divby)
        ns = Int64.(ceil.(d2[1:3]./dn))
        getsc(d,dn,ns) = (ns==1 ? 0f0 : (d-dn)/(ns-1))
        sc = getsc.(d2[1:3],dn,ns)
        #print(dn," ", ns, " ", sc, "\n")
        for k=1:ns[3]
            for j=1:ns[2]
                for i=1:ns[1]
                    ii = ((1:dn[1]).+Int64(round((i-1)*sc[1])),
                          (1:dn[2]).+Int64(round((j-1)*sc[2])),
                          (1:dn[3]).+Int64(round((k-1)*sc[3])))

                    Y[ii...,:,:] .+= net(X[ii...,:,:])
                    W[ii...] .+= 1
                    print(".")
                end
            end
        end
        Y .= Flux.softmax(Y./W, dims=4)
    else
        function reflect(i, d)
            j = mod(i-1, 2d)
            b = UInt64((j<d) ? j+1 : 2d-j)
        end

        dn = Int64.(ceil.(d2[1:3]./divby).*divby)
        t  = Int64.(ceil.((dn[1:3] .- d2[1:3])./2))
        x  = similar(X,(dn...,size(X,4),1))
        x[1:dn[1],1:dn[2],1:dn[3],:,1] = X[reflect.((1:dn[1]).-t[1],d2[1]),
                                           reflect.((1:dn[2]).-t[2],d2[2]),
                                           reflect.((1:dn[3]).-t[3],d2[3]),:,1]
        y  = net(x)
        Y  = Flux.softmax(y[(1:d2[1]).+t[1], (1:d2[2]).+t[2], (1:d2[3]).+t[3],:,1],
                          dims=4)
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

