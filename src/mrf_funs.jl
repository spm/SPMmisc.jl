function sum_neighbours(f,w,i,j,k)
   #t = (f[i-1,j,k] + f[i+1,j,k] + f[i,j-1,k] + f[i,j+1,k] + f[i,j,k-1] + f[i,j,k+1])/6f0
    t = (f[i-1,j,k].*w[i-1,j,k] + f[i+1,j,k].*w[i+1,j,k] + f[i,j-1,k].*w[i,j-1,k] + f[i,j+1,k].*w[i,j+1,k] +
         f[i,j,k-1].*w[i,j,k-1] + f[i,j,k+1].*w[i,j,k+1]) / 
        (w[i-1,j,k] + w[i+1,j,k] + w[i,j-1,k] + w[i,j+1,k] + w[i,j,k-1] + w[i,j,k+1] + 1f-4)
end

function mrf(PG,PW,PB,S0,lam1=1f0,lam2=1f0)
    d = size(PG)
    G = deepcopy(PG)
    W = deepcopy(PW)
    B = deepcopy(PB)
    S = 1 .- S0
    for it=1:50
        for k=2:(d[3]-1)
            for j=2:(d[2]-1)
                istart = ((it+j+k) % 2) + 2
                for i=istart:2:(d[1]-1)
                    # Could be made much more efficient
                    sg = sum_neighbours(G,S,i,j,k)
                    sw = sum_neighbours(W,S,i,j,k)
                    sb = sum_neighbours(B,S,i,j,k)
                    g = PG[i,j,k] * exp(lam1*sg) + 1f-6
                    w = PW[i,j,k] * exp(lam1*sw-lam2*sb) + 1f-6
                    b = PB[i,j,k] * exp(lam1*sb-lam2*sw) + 1f-6
                    s = g + w + b 
                    G[i,j,k] = g / s
                    W[i,j,k] = w / s
                    B[i,j,k] = b / s
                end
            end
        end
        print(".")
    end
    print("\n")
    return G,W,B
end

