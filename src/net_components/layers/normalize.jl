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
    println("Applying Normalize layer")
    println("Size of mean, std, x: $(size(p.mean)), $(size(p.std)), $(size(x))")
    # @assert size(p.mean) == size(p.std) == size(x)
    output = (x.-p.mean)./p.std
    return output
end

(p::Normalize)(x::Array{<:JuMPReal}) = apply(p, x)
