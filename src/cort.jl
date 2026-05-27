using NIfTI
nii = NIfTI
include("thin.jl")
include("lapl.jl")
include("morphological.jl")
include("mrf_funs.jl")
include("skeletonize.jl")


function write_vol(f, fname, h1; descrip="")
    ovol = nii.NIVolume(Float32.(f))
    ovol.header.bitpix     = 4
    ovol.header.datatype   = 16
    ovol.header.scl_slope  = 1
    ovol.header.scl_inter  = 0
    ovol.header.pixdim     = h1.header.pixdim
    ovol.header.qform_code = h1.header.qform_code
    ovol.header.qoffset_x  = h1.header.qoffset_x
    ovol.header.qoffset_y  = h1.header.qoffset_y
    ovol.header.qoffset_z  = h1.header.qoffset_z
    ovol.header.quatern_b  = h1.header.quatern_b
    ovol.header.quatern_c  = h1.header.quatern_c
    ovol.header.quatern_d  = h1.header.quatern_d
    ovol.header.sform_code = h1.header.sform_code
    #nii.setaffine(ovol.header, M1)
    ovol.header.srow_x     = h1.header.srow_x
    ovol.header.srow_y     = h1.header.srow_y
    ovol.header.srow_z     = h1.header.srow_z
    ovol.header.descrip    = nii.string_tuple(descrip, 80)
    ovol.header.xyzt_units = h1.header.xyzt_units
    nii.niwrite(fname,ovol)
end

function prepend(fname,prefix)
    pth = splitpath(fname)
    pth[end] = prefix * pth[end]
    joinpath(pth)
end


P1="c1IXI226-HH-1618-T1.nii"
P2="c2IXI226-HH-1618-T1.nii"

P1="c1IXI226-HH-1618-T2.nii"
P2="c2IXI226-HH-1618-T2.nii"

#P1 = "c1IXI017-Guys-0698-T1.nii"
#P2 = "c2IXI017-Guys-0698-T1.nii"

P1 = "cortex_IXI013-HH-1212.nii"
P2 = "intern_IXI013-HH-1212.nii"

P1 = "cortex_IXI002-Guys-0828.nii"
P2 = "intern_IXI002-Guys-0828.nii"

function cort(P1,P2)

h1 = niread(P1)
h2 = niread(P2)
M1 = nii.getaffine(h1)
M2 = nii.getaffine(h2)
vx = sqrt.(sum(M1[1:3,1:3].^2,dims=1))

p1 = Float32.(h1.raw).*h1.header.scl_slope
p2 = Float32.(h2.raw).*h2.header.scl_slope
p3  = max.(1 .- p1 .- p2,0)

nerode   = 1
ncdilate = 16

print("Erode (",nerode,") & conditional dilate (", ncdilate, ") GM + WM\n")
f  = erode(p1.+p2,nerode)
f  = cond_dilate!(f, p1.+p2, ncdilate)
p1 = min.(f,p1)
p2 = min.(f,p2)
p3 = max.(1 .- p1 .- p2,0)

print("Erode (",nerode,") & conditional dilate (", ncdilate, ") WM\n")
f  = erode(p2,nerode)
f  = cond_dilate!(f, p2, ncdilate)
p1 = min.(1 .- f, p1)
p3 = min.(1 .- f, p3)
p2 = max.(1 .- p1 .- p3,0)

function mean_skel(p,p0=zero(p),nt=8)
    print("Average skeleton at ", nt, " different thresholds\n")
    f  = zero(p)
    thresholds = (1/nt/2):(1/nt):(1-1/nt/2)
    for t=thresholds
        print(t," ")
        f .+= skeletonize(p .> t, p0; nit=10 )
    end
    f ./= length(thresholds)
end

function mean_skel1(p, p0=zero(p), nsamp=8)
    # This option does not work so well in practice because
    # random noise complicates the topology
    print("Average skeleton of ", nsamp, " samples\n0 ")
    f  = Float32.(skeletonize(p .> 0.5, p0; nit=10))
    for t=1:(nsamp-1)
        print(t," ")
        b   = p .> rand(Float32,size(p))
        f .+= skeletonize(b, p0; nit=10)
    end
    f ./= nsamp
end

bskel = mean_skel(p1.+p3, p3, 8)

write_vol(bskel,prepend(P1,"skel-"), h1; descrip="Skeleton")

p3  = max.(p3, bskel) 
p1  = max.(1 .- p2 .- p3,0)


print("MRF: ")
(p1, p2, p3) = mrf(p1,p2,p3,bskel, 1f0,12f0)
p3  = max.(p3, bskel)
p1  = max.(1 .- p2 .- p3,0)


write_vol(p1,prepend(P1,"clean-"), h1; descrip="GM")
write_vol(p2,prepend(P2,"clean-"), h1; descrip="WM")

print("Laplacian: ")
f = lapl(p2,p3)
f[:,:,1].=0; f[:,:,end].=0
f[:,1,:].=0; f[:,end,:].=0
f[1,:,:].=0; f[end,:,:].=0

write_vol(f, prepend(P1,"depth-"), h1; descrip="Relative depth")
end

