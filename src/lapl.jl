function lapl(w,b)
    d = size(w)
    f = zero(w) # .+ 0.5 # max.(w, min.(1 .- b,0.5f0))
    for it=1:100
        for k=2:(d[3]-1)
            for j=2:(d[2]-1)
                istart = ((it+j+k) % 2) + 2
                for i=istart:2:(d[1]-1)
                    t        = (f[i-1,j,k] + f[i+1,j,k] + f[i,j-1,k] + f[i,j+1,k] + f[i,j,k-1] + f[i,j,k+1])/6f0
                    f[i,j,k] = max(w[i,j,k], min(1-b[i,j,k], t))
                end
            end
        end
        print(".")
    end
    print("\n")
    f
end

