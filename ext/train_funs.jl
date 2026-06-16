using Zygote, Flux, CUDA, JLD2
using NIfTI
using Statistics, Random
using Printf, JLD2

function nparam(net)
    sum(length.(Flux.trainables(net)))
end

function train!(net, data, state, settings, sett, nam, eta)
    convert = CuArray
    losses  = zeros(settings.epochs)
    theta   = Flux.trainables(net)

    snet    = deepcopy(net)
    stheta  = Flux.trainables(snet)
    for i=1:length(theta)
        stheta[i] .= theta[i]
    end
    for epoch in 1:settings.epochs
        loss = 0
        ns   = 0
        bn   = 0
        nr   = 0
        lr   = 0
        t0   = time()
        Optimisers.adjust(state, eta = eta, delta = eta)
        while true
            x,y = next!(data)
            if x==nothing || y==nothing
                break
            else
                x   = convert(x)
                y   = convert(y)
                ns += 1

                train_loss, back = Zygote.pullback(model -> settings.loss(model(x), y), net)
                grads   = back(one(train_loss))

                state, net = Optimisers.update!(state, net, grads[1])
                loss += train_loss
                bn   += 1
                #if(rem(bn,10)==0); print("▩"); flush(stdout); end
                nr   += 1
                lr   += train_loss

                for i=1:length(theta)
                    stheta[i] .= stheta[i].*Float32.((nr-1)/nr) + theta[i].*Float32.(1/nr)
                end

                if(rem(bn,16)==0)
                    @printf(stdout, "%3d ", round(1000*lr/nr)); flush(stdout);
                    nr = 0
                    lr = 0f0
                end
            end
        end
        losses[epoch] = loss/bn
        #td = round((time()-t0)*10)/10
        savenet(cpu(net),  "runs/"*nam*".jld2", sett)
        savenet(cpu(snet), "runs/smo-"*nam*".jld2", sett)

    end
    losses[end], snet
end


function checkit(state,theta,grads)
    for n=1:length(theta)
        if any(.~isfinite.(theta[n]))
            print("theta[",n,"] is a problem.\n")
            theta = cpu(theta)
            state = cpu(state)
            grads = cpu(grads)
            @save "state.jld2" state theta grads
        end
        nothing
    end
end


function test(net, P, lossfun)
    Loss = zeros(length(P[1][1]))
   #convert = isa(Flux.trainables(net)[1], CuArray) ? CuArray : (x)->Float32.(x)
    convert = CuArray
   #convert = identity
    for n in 1:length(P[1][1])
        ns   = nslices(P[1][1][n])
        loss = 0
        for i in 1:ns
            local x = convert(get_slices(P[1],n,i))
            local y = convert(get_slices(P[2],n,i))
            y1      = net(x)
            tmp     = cpu(lossfun(y1, y))
            loss   += tmp
        end
        Loss[n] = loss/ns
       #print(round(1000*(Loss[n]))/1000, " ")
    end
    #print("\n", mean(Loss),"\n")
    mean(Loss)
end

function conf!(M,y0,y1)
    for j=1:size(y0,2), i=1:size(y0,1)
        l0 = argmax(view(y0,i,j,:))
        l1 = argmax(view(y1,i,j,:))
        M[l0,l1] += 1
    end
    return M
end

function DSC(M)
    a = zeros(size(M,1))
    for i=1:length(a)
        a[i] = 2*M[i,i]/(sum(M[i,:]) + sum(M[:,i]))
    end
    return a
end

function test_dsc(net, P, lossfun)
    d3   = length(P[2])+1
    N    = length(P[2][1])
    Loss = zeros(N)
    D    = zeros(d3,N)
   #convert = isa(Flux.trainables(net)[1], CuArray) ? CuArray : (x)->Float32.(x)
    convert = CuArray
   #convert = identity
    for n in 1:N
        ns   = nslices(P[1][1][n])
        loss = 0
        M    = zeros(d3,d3)
        for i in 1:ns
            local x = convert(get_slices(P[1],n,i))
            local y = convert(get_slices(P[2],n,i))
            x = reshape(x,(size(x)[1:3]...,1))
            y = reshape(y,(size(y)[1:3]...,1))
            y1      = net(x)
            tmp     = cpu(lossfun(y1, y))
            loss   += tmp
            conf!(M,cpu(y1),cpu(y))
        end
        Loss[n] = loss/ns
        D[:,n]  = DSC(M)
        #print(round(1000*(Loss[n]))/1000, " ")
    end
    #print("\n", mean(Loss),"\n")
    md = sum(D,dims=2)/N
    return mean(Loss), md
end


