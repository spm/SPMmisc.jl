function get_cort(ind=[])
    dr = "/home/john/cortex/seg"
    f0 = readdir(dr,join=true,sort=true)
    t1 = occursin.(Regex("/IXI.*T1.nii"),f0)
    t2 = occursin.(Regex("/clean-cortex_IXI.*.nii"),f0)
    t3 = occursin.(Regex("/clean-intern_IXI.*.nii"),f0)
    f1 = f0[t1]
    f2 = f0[t2]
    f3 = f0[t3]
    if isempty(ind)
        ind = 1:length(f1)
    end
    return ((f1[ind],),(f2[ind],f3[ind]))
end

function get_sim(ind=[])
    dr = "/home/john/cortex/sim"
    f0 = readdir(dr,join=true,sort=true)
    t1 = occursin.(Regex("/IXI.*.nii"),f0)
    t2 = occursin.(Regex("/c1IXI.*.nii"),f0)
    t3 = occursin.(Regex("/c2IXI.*.nii"),f0)
    f1 = f0[t1]
    f2 = f0[t2]
    f3 = f0[t3]
    if isempty(ind)
        ind = 1:length(f1)
    end
    return ((f1[ind],),(f2[ind],f3[ind]))
end

function get_sim_gwc(ind=[])
    dr0 = "/home/john/cortex/sim"
    dr1 = "/home/john/cortex/seg"
    f0 = readdir(dr0,join=false,sort=true)
    t1 = occursin.(Regex("^IXI.*.nii"),f0)
    f1 = f0[t1]
    if isempty(ind)
        ind = 1:length(f1)
    end
    f1 = f1[ind]
    f0 = Vector{String}(undef,length(f1))
    c1 = Vector{String}(undef,length(f1))
    c2 = Vector{String}(undef,length(f1))
    c3 = Vector{String}(undef,length(f1))
    for n=1:length(f1)
        ff = split(f1[n],".nii")[1]
        f0[n] = dr0 * "/" * ff * ".nii"
        c1[n] = dr1 * "/" * "c1" * ff * "-T1.nii"
        c2[n] = dr1 * "/" * "c2" * ff * "-T1.nii"
        c3[n] = dr1 * "/" * "c3" * ff * "-T1.nii"
    end
    return ((f0,),(c1,c2,c3))
end


tr_data   = get_sim_gwc(1:60);

sm(x)     = softmax(x,dims=ndims(x)-1)
lossfun   = (yₓ,y) -> Flux.Losses.logitcrossentropy(yₓ, y; agg=mean, dims=4)
batchsize = 1


function copy_params!(theta_new, theta)
    for i=1:length(theta)
        theta_new[i] .= theta[i]
    end
end


nam = "Classic-BUnet3"

Random.seed!(123)
CUDA.seed!(123)
network = "runs/"*nam*".jld2"

if true
    sett     = (([1, 25, 50, 100, 200, 200],),(cout=4,),pBUnet3)
    net      = sett[3](sett[1]...; sett[2]...)
else
    net,sett = loadnet(network)
end
net      = gpu(net)

data     = VolData(tr_data[1], tr_data[2])

ofile = "csv/"*nam*".csv"
print(ofile, "\n"); flush(stdout)
io    = open(ofile, write=true)

eta0  = 0.0002
eta_d = 0.99
eta   = eta0 * (eta_d^0)
opt   = Tadam(eta, (0.9,0.999), 1e-8)
state = Optimisers.setup(opt,net)

for epoch=1:2000
    eta = eta*eta_d
    tr_loss, snet = train!(net, data, state, (epochs = 1, loss = lossfun), sett, nam, eta)
    @printf(stdout, "   %4d, %g\n", epoch, tr_loss)
    flush(stdout)
    @printf(io, "%4d, %g\n", epoch, tr_loss)
    flush(io)
end
close(io)
net

