using Flux

function skeletonize(B::AbstractArray{Bool}, G=zero(B); nit=1024)
    D        = size(B)
    F        = ones(UInt8,3,3,3)
    F[2,2,2] = 0
    r        = 0:2
    w        = 2 .^[Array(0:12)..., 0, Array(13:25)...]
    w[14]    = 0
    lkp      = get_lookup()

    F        = reshape(F,(size(F)...,1,1))
    o        = CartesianIndex(1,1,1)
    c        = zeros(Bool,3,3,3)

    Bview    = view(B, 2:(D[1]-1), 2:(D[2]-1), 2:(D[3]-1))
    Gview    = view(G, 2:(D[1]-1), 2:(D[2]-1), 2:(D[3]-1))

    nit      = min(nit,minimum(size(B)))
    for it=1:nit 
        removed_it = 0
        C0         = reshape(conv(UInt8.(reshape(B,(size(B)...,1,1))),F),(D[1]-2,D[2]-2,D[3]-2))

        # Scan edge voxels, first removing voxels with fewer neighbours
        for thresh = 8:26

            # Candidate voxels for removal, prioritising those with lower probability
            # of being CSF.
            candidates = findall( (Bview.==1) .& (C0 .+ Gview.*(nit-it) .<thresh) ) .+ o

            for subit=1:32                          # Iterate until no more voxels can be removed
                candidates    = reverse(candidates) # Reverse the directions at each sub-iteration
                removed_subit = 0                   # Count of removed voxels in this subiteration
                for ii=candidates                   # Loop over candidate voxels
                    if B[ii] 
                        patch      = view(B,(ii.-o):(ii.+o)) # Neighbouring 3x3x3 patch
                        neighbours = sum(patch) - 1          # Count number of neighbours
                        if neighbours > 8 || neighbours < 5
                            # Not entirely sure of the best values to use regarding the
                            # number of neighbours to use. A complete surface should have
                            # eight neighbours, whereas I think the edge should have about
                            # five. If a point has fewer than five neighbours, then I think
                            # we can assume that it is not part of a surface.
                            # 
                            # Mid-surface   Edge
                            #    * * *      . . .
                            #    * o *      * o *
                            #    * * *      * * *

                            # Convert the pattern of 0s and 1s to an integer and read the
                            # lookup table at this value
                            if ~lkp[w'*patch[:].+1]
                                # If the lookup table says flipping from 1 to 0 leaves the
                                # topology unchanged, then set to 0
                                removed_subit += 1
                                B[ii]          = false
                            end
                        end
                    end
                end
                removed_it += removed_subit
                if removed_subit == 0
                    break
                end
            end
        end
        print(".")

        if removed_it == 0
            # Done
            break
        end
    end
    print("\n")
    B
end

function get_lookup()
    file = joinpath(@__DIR__, "topology_lookup.dat")
    fp   = open(file,"r")
    n    = Int64((2^26)/8);
    lu8  = read(fp,n)
    close(fp)
    lu   = zeros(Bool,8,n)
    for i=1:8
        msk = 0x2^(i-1)
        lu[i,:] .= (lu8 .& msk) .== msk
    end
    lu = lu[:]
end

