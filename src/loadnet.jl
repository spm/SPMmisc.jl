using JLD2, Flux

"""
    loadnet(filename)

Load a neural network from a file.
Network architectures are described by some function that must
be avaiable at the time of loading.
"""
function loadnet(filename)
    print("Loading ", filename)
    sett = JLD2.load(filename, "settings")
    net  = sett[3](sett[1]...; sett[2]...)
    W1   = JLD2.load(filename, "W")
    W0   = Flux.trainables(net)
    for i=1:length(W0)
        W0[i] .= W1[i]
    end
    print("\n")
    return net, sett
end

function savenet(net, filename, sett)
    W = Flux.trainables(net)
    JLD2.jldsave(filename; W = W, settings=sett)
end

