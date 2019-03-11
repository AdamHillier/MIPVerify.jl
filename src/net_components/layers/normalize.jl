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
    print(io, "Normalize(size: $(size(p.mean)))")
end

function apply(p::Normalize, x::Array{<:JuMPReal})
    padded_shape = (ones(Int, ndims(x) - ndims(p.mean))..., size(p.mean)...)
    m = reshape(p.mean, padded_shape)
    s = reshape(p.mean, padded_shape)
    return (x .- m) ./ s
end

(p::Normalize)(x::Array{<:JuMPReal}) = apply(p, x)
