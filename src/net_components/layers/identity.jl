export Identity

"""
$(TYPEDEF)

Is just an identity layer.
"""
struct Identity <: Layer end

Base.hash(a::Identity, h::UInt) = hash(:Identity, h)

function Base.show(io::IO, p::Identity)
    print(io, "Identity()")
end

(p::Identity)(x::Array{<:JuMPReal}) = x
