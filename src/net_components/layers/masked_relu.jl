export MaskedReLU

"""
$(TYPEDEF)

Represents a masked ReLU activation, with `mask` controlling how the ReLU is applied to
each output.

`p(x)` is shorthand for [`masked_relu(x, p.mask)`](@ref) when `p` is an instance of
`MaskedReLU`.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct MaskedReLU{T<:Real} <: Layer
    mask::Array{T}
    tightening_algorithm::Nullable{TighteningAlgorithm}
end

function MaskedReLU(mask::Array{T}) where {T<:Real}
    MaskedReLU{T}(mask, Nullable{TighteningAlgorithm}())
end

function MaskedReLU(mask::Array{T}, ta::TighteningAlgorithm) where {T<:Real}
    MaskedReLU{T}(mask, Nullable{TighteningAlgorithm}(ta))
end

function Base.show(io::IO, p::MaskedReLU)
    num_zeroed_units = count(p.mask .< 0)
    num_passthrough_units = count(p.mask .> 0)
    num_rectified_units = length(p.mask) - num_zeroed_units - num_passthrough_units
    print(io,
        "MaskedReLU with expected input size $(size(p.mask)). ($(num_zeroed_units) zeroed, $(num_passthrough_units) as-is, $(num_rectified_units) rectified)."
    )
end

function apply(p::MaskedReLU, x::Array{<:Real})
    padded_shape = (ones(Int, ndims(x) - ndims(p.mask))..., size(p.mask)...)
    m = reshape(p.mask, padded_shape)
    masked_relu(x, m)
end

function apply(p::MaskedReLU, x::Array{<:JuMPLinearType})
    info(MIPVerify.LOGGER, "Applying $p ... ")
    padded_shape = (ones(Int, ndims(x) - ndims(p.mask))..., size(p.mask)...)
    m = reshape(p.mask, padded_shape)
    masked_relu(x, m, nta = p.tightening_algorithm)
end

(p::MaskedReLU)(x::Array{<:JuMPReal}) = apply(p, x)
