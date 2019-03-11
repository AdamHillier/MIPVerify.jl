export Normalize

"""
$(TYPEDEF)

Represents a Normalization operation.
"""
@auto_hash_equals struct Normalize <: Layer
    mean::Array{Real}
    std::Array{Real}
end

function Base.show(io::IO, p::Normalize)
    print(io, "Normalize(means: $(p.mean), stds: $(p.std))")
end

function apply(p::Normalize, x::Array{<:JuMPReal})
    @assert size(p.mean) == size(p.std) == size(x)
    output = (x.-p.mean)./p.std
    return output
end

(p::Normalize)(x::Array{<:JuMPReal}) = apply(p, x)
