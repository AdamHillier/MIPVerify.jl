export Graph

"""
$(TYPEDEF)
Represents a generic (feed-forward) neural net, with `layers` which can take
inputs from any of the preceeding layers. This allows skip connections, etc.,
but naturally disallows cycles.
## Fields:
$(FIELDS)
"""
@auto_hash_equals struct Graph <: NeuralNet
    layers::Array{Layer, 1} # Array of layers
    inputs::Dict{Int, Array{Int, 1}} # Look-up table for input indices
    UUID::String
end

function Base.show(io::IO, p::Graph)
    println(io, "graph net $(p.UUID)")
    for (index, layer) in enumerate(p.layers)
        println(io, "  ($(get(p.inputs, index, []))) $layer")
    end
end

# TODO (vtjeng): Think about the types carefully.
function apply(p::Graph, x::Array{<:JuMPReal})
    # Every layer must have at least one input
    for i in 1:length(p.layers)
        @assert haskey(p.inputs, i) && length(p.inputs[i]) > 0
    end

    outputs_cache = Dict{Int, Array{<:JuMPReal}}()

    function get_output(index)
        println("Get output index $index")
        if index == 0
            println("Returning x")
            return x
        elseif haskey(outputs_cache, index)
            println("Returning cached value")
            return outputs_cache[index]
        else
            println("Recursive step")
            input_values = map(get_output, p.inputs[index])
            println("Saving values to cache")
            outputs_cache[index] = p.layers[index](input_values...)
            return outputs_cache[index]
        end
    end

    return get_output(length(p.layers))
end

(p::Graph)(x::Array{<:JuMPReal}) = apply(p, x)
