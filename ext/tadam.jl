import Optimisers.apply!, Optimisers.init, Optimisers.@def,  Optimisers.@lazy, Optimisers.AbstractRule

"""

    Tadam(η, η₀ = η, β = (0.9, 0.999))

Adam with a tanh

  Parameters
  ----------
  * `η`  - Update step size
  * `β`  - Decay parameters, similar to those of Adam.
         - `β[1]` - - A faster decay for computing smoothed means of gradients ("momentum"). This parameter is related to an effective batch-size of the training.
         - `β[2]` - A slower dacay for computing moments for the Hessian estimates and mean of the parameters. This parameter should be related to the effective size of the training dataset (accounting for augmentations), as 1-1/N.
  * `ϵ`  - For numerical stability (preventing 0/0). Plays a similar role to the ϵ in Adam.
"""
@def struct Tadam <: AbstractRule
    eta     = 0.001
    beta    = (0.9,0.999)
    epsilon = 1e-8
end

init(o::Tadam, x::AbstractArray{T}) where T = (Float64.([0, o.epsilon]), zero(x), zero(x), zero(x))


function apply!(o::Tadam, state, x::AbstractArray{T}, Δ) where T
    η, β₁, β₂, ϵ = T(o.eta), T(o.beta[1]), T(o.beta[2]), T(o.epsilon)
    Σ1, VΔΔ, ΣΔ, ΣΔ₁ = state

    nd     = Σ1[2]
    w      = nextfloat(Float32(β₂ * nd/(β₂*nd + 1)))
    n      = T(nd)
    @. VΔΔ = β₂ * VΔΔ + w*(abs(Δ - ΣΔ/n)+2eps(Δ))^2
    nd     = β₂ * nd + 1
    Σ1[2]  = nd
    n      = T(nd)

    @. ΣΔ  = β₂ * ΣΔ  + Δ

    m      = β₁ * Σ1[1] + 1
    Σ1[1]  = m
    @. ΣΔ₁ = β₁ * ΣΔ₁   + Δ


    # Compute update
    Δ′ = @lazy η*tanh((1/m)*ΣΔ₁/√((VΔΔ + ΣΔ*ΣΔ/n)/n + ϵ))

    return (Σ1, VΔΔ, ΣΔ, ΣΔ₁), Δ′
end


function Base.show(io::IO, o::Tadam)
    print(io, "Tadam(eta=$(o.eta), beta=$(o.beta), epsilon=$(o.epsilon))")
end

