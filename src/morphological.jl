
function erode!(E,P)
    n = size(P)
    for k=2:(n[3]-1)
        for j=2:(n[2]-1)
            for i=2:(n[1]-1)
                E[i,j,k] = min(P[i,j,k],P[i-1,j,k],P[i+1,j,k],P[i,j-1,k],P[i,j+1,k],P[i,j,k-1],P[i,j,k+1])
            end
        end
    end
end

function dilate!(E,P)
    n = size(P)
    for k=2:(n[3]-1)
        for j=2:(n[2]-1)
            for i=2:(n[1]-1)
                E[i,j,k] = max(P[i,j,k],P[i-1,j,k],P[i+1,j,k],P[i,j-1,k],P[i,j+1,k],P[i,j,k-1],P[i,j,k+1])
            end
        end
    end
end

function erode(P,n=1)
    E1 = zero(P)
    E0 = deepcopy(P)
    for i=1:n
        erode!(E1,E0)
        E0 .= E1
    end
    E1
end

function dilate(P,n=1)
    E1 = zero(P)
    E0 = deepcopy(P)
    for i=1:n
        dilate!(E1,E0)
        E0 .= E1
    end
    E1
end

function cond_dilate!(E,P,n=1)
    E1 = zero(P)
    for i=1:n
        dilate!(E1,E)
        E .= min.(E1,P)
    end
    E
end

function cond_erode!(E,P,n=1)
    E1 = zero(P)
    for i=1:n
        erode!(E1,E)
        E .= max.(E1,P)
    end
    E
end

