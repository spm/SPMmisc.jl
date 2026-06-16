using PushPull
pp = PushPull

"""
    uniform_random_rotation3d()

Return a 4x4 affine transformation that encodes a uniformly
random rotation.

See https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/

Note that augmenting this way seems to requires a larger network and takes longer
to train. It may be better to assume that we know the approximate head orientation
from the sform/qform in the NIfTI headers, and limit the amount of augmentation.
"""
function uniform_random_rotation3d()
    x₁ = rand()
    c  = cos(x₁*2π)
    s  = sin(x₁*2π)
    R  = [ c s 0
          -s c 0
           0 0 1]
    x₂ = rand()
    x₃ = rand()
    v  = [cos(x₂*2π)*sqrt(x₃)
          sin(x₂*2π)*sqrt(x₃)
          sqrt(1-x₃)];
    H  = one(R) - 2*v*v' return [-H*R [0,0,0]; 0 0 0 1]
end

"""
    eye(N)

The eye() function in MATLAB is often useful.
"""
function eye(N::Integer=4)
    M = zeros(Float32,N,N)
    for n=1:N
        M[n,n] = 1f0
    end
    return M
end

"""
    translate_matrix(t::Vector{<:AbstractFloat}=[0f0, 0f0, 0f0])

4x4 translation matrix.
"""
function translate_matrix(t::Vector{<:AbstractFloat}=[0f0, 0f0, 0f0])
    M = eye(4)
    for n=1:3
        M[n,4] = t[n]
    end
    return M
end


function random_affine_transform(zoom_sd = 0)
    R = uniform_random_rotation3d()
    if zoom_sd != 0
        Rz = uniform_random_rotation3d()
        z  = randn(Float32,3) .*= zoom_sd
        Z  = Rz' * [exp(z[1]) 0f0 0f0 0f0;
                    0f0 exp(z[2]) 0f0 0f0;
                    0f0 0f0 exp(z[3]) 0f0;
                    0f0 0f0 0f0       1f0] * Rz
        R  = R*Z
    end
    T = translate_matrix(rand(Float32,3) .*= 1000f0)
    R = R*T
    return R
end


function random_patch(dataX, dataY, dim, mat=nothing; zoom_sd = 0)
    sett = PushPull.Settings((1,1,1), (1,1,1), 1)
    X    = zeros(Float32,(dim[1:3]...,length(dataX),dim[4]))
    Y    = zeros(Float32,(dim[1:3]...,length(dataY),dim[4]))
    for b=1:dim[end]
        R = random_affine_transform(0.1f0)
        for c=1:length(dataX)
            M = Float32.(dataX[c].mat\R)
            X[:,:,:,c,b] = affine_pull(dataX[c].volume,M,dim[1:3],sett)
        end
        for c=1:length(dataY)
            M = Float32.(dataY[c].mat\R)
            Y[:,:,:,c,b] = affine_pull(dataY[c].volume,M,dim[1:3],sett)
        end
    end
    X[isnan.(X)] .= 0
    Y[isnan.(Y)] .= 0
    return X,Y
end

"""
    rand_contrast(x; contrast_sd=0.02f0, tiny=1f-3)

Take a set of input images and combine them randomly in log-space.

Note that the code is slightly different for the case of three
input images, where it assumes that they are T1-weighted, MT-weighted
and T2-weighted. This is so that the T1-weighted has a slightly bigger
influence because this is what most people seem to be using.
"""
function rand_contrast(x; contrast_sd=0.02f0, tiny=1f-3)
    nc = size(x,4)
    if nc>1
        if nc==3
            r1 = rand()
            r2 = rand()
            r  = [(1-r1), r1*r2, r1*(1-r2)] .+ randn()*contrast_sd
        else
            r  = rand(Float32,nc)
            r  = r./sum(r) .+ randn(1)*contrast_sd
        end
        x1 = zero(x[:,:,:,1:1,:])
        for c=1:nc
            x1 .+= log.(max.(x[:,:,:,c:c,:],tiny)).*r[c]
        end
        x1 = exp.(x1)
    end
    return x1
end

"""
    rand_inu(x; rescale_sd = 1f0)

Augment by adding random intensity nonuniformity.

This is currently done using the exponential of a relatively
simple polynomial function.

"""
function rand_inu(x; rescale_sd = 1f0)
    if rescale_sd != 0
        d  = size(x)
        f1 = (Array(1:d[1]) .- d[1]/2)./d[1]
        f2 = (Array(1:d[2]) .- d[2]/2)./d[2]
        f3 = (Array(1:d[3]) .- d[3]/2)./d[3]
        F1 = [f1.^0 f1.^1 f1.^2 f1.^3]
        F2 = [f2.^0 f2.^1 f2.^2 f2.^3]
        F3 = [f3.^0 f3.^1 f3.^2 f3.^3]
        F  = zero(x)
        R  = randn(Float32,4,4,4).*rescale_sd
        for i=1:size(F,3)
            r = reshape(F3[i:i,:]*reshape(R,(4,16)),(4,4))
            F[:,:,i,1,1] .= F1*r*F2'
        end
        x .*= Float32.(exp.(F))
    end
end


"""
    rand_ncchi(x; noise_sd=8f0, maxchan=2)

Augment with noncentral chi noise, where the number
of channes is uniformly distributed between 2 and maxchan.
"""
function rand_ncchi(x; noise_sd=8f0, maxchan=2)
    e  = zero(x)
    nc = 1 + ceil((maxchan-1)*rand())
    for i=1:nc
        e .+= randn(Float32,size(e)).^2
    end
    x .= sqrt.(x.^2 .+ (randn()*noise_sd).^2 .* e)
end


function augment_seg(x,y; rescale_sd=1f0, contrast_sd=0.05f0, noise_sd=8f0, maxchan=16)

    # For categorical target data (i.e. segmentation), where
    # a background class is needed.
    y = cat(y,max.(1f0 .- sum(y,dims=4),0f0), dims=4)

    # Various image augmentations
    x = rand_contrast(x; contrast_sd=contrast_sd)
    x = rand_inu(x; rescale_sd=rescale_sd)
    x = rand_ncchi(x; noise_sd=noise_sd, maxchan=maxchan)
    return x,y
end

