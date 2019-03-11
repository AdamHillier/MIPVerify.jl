export Add

"""
$(TYPEDEF)

Output the sum of two inputs.
"""
struct Add <: Layer end

Base.hash(a::Add, h::UInt) = hash(:Add, h)

function Base.show(io::IO, p::Add)
    print(io, "Add()")
end

(p::Add)(x::Array{<:JuMPReal}, y::Array{<:JuMPReal}) = x + y
