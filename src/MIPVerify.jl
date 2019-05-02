module MIPVerify

using Base.Cartesian
using JuMP
using ConditionalJuMP
using Memento
using AutoHashEquals
using DocStringExtensions
using ProgressMeter
using CSV
using DataFrames

const dependencies_path = joinpath(Pkg.dir("MIPVerify"), "deps")

export find_adversarial_example, frac_correct, interval_arithmetic, lp, mip

@enum TighteningAlgorithm interval_arithmetic=1 lp=2 mip=3
@enum AdversarialExampleObjective closest=1 worst=2
const DEFAULT_TIGHTENING_ALGORITHM = mip

abstract type AdversarialPredictionOption end
type None <: AdversarialPredictionOption end
struct MinDistanceThreshold <: AdversarialPredictionOption
    threshold::Float32
end
type FindMinDistanceThreshold <: AdversarialPredictionOption end

include("net_components.jl")
include("models.jl")
include("utils.jl")
include("logging.jl")

function get_max_index(
    x::Array{<:Real, 1})::Integer
    return findmax(x)[2]
end

function get_default_tightening_solver(
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver
    )::MathProgBase.SolverInterface.AbstractMathProgSolver
    tightening_solver = typeof(main_solver)()
    MathProgBase.setparameters!(tightening_solver, Silent = true, TimeLimit = 20)
    return tightening_solver
end

"""
$(SIGNATURES)

Finds the perturbed image closest to `input` such that the network described by `nn`
classifies the perturbed image in one of the categories identified by the 
indexes in `target_selection`.

`main_solver` specifies the solver used to solve the MIP problem once it has been built.

The output dictionary has keys `:Model, :PerturbationFamily, :TargetIndexes, :SolveStatus,
:Perturbation, :PerturbedInput, :Output`. 
See the [tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/03_interpreting_the_output_of_find_adversarial_example.ipynb)
on what individual dictionary entries correspond to.

*Formal Definition*: If there are a total of `n` categories, the (perturbed) output vector 
`y=d[:Output]=d[:PerturbedInput] |> nn` has length `n`. 
We guarantee that `y[j] - y[i] ≥ tolerance` for some `j ∈ target_selection` and for all `i ∉ target_selection`.

# Named Arguments:
+ `invert_target_selection::Bool`: Defaults to `false`. If `true`, sets `target_selection` to 
    be its complement.
+ `pp::PerturbationFamily`: Defaults to `UnrestrictedPerturbationFamily()`. Determines
    the family of perturbations over which we are searching for adversarial examples.
+ `norm_order::Real`: Defaults to `1`. Determines the distance norm used to determine the 
    distance from the perturbed image to the original. Supported options are `1`, `Inf` 
    and `2` (if the `main_solver` used can solve MIQPs.)
+ `tolerance::Real`: Defaults to `0.0`. See formal definition above.
+ `rebuild::Bool`: Defaults to `false`. If `true`, rebuilds model by determining upper and lower
    bounds on input to each non-linear unit even if a cached model exists.
+ `tightening_algorithm::MIPVerify.TighteningAlgorithm`: Defaults to `mip`. Determines how we
    determine the upper and lower bounds on input to each nonlinear unit. 
    Allowed options are `interval_arithmetic`, `lp`, `mip`.
    (1) `interval_arithmetic` looks at the bounds on the output to the previous layer.
    (2) `lp` solves an `lp` corresponding to the `mip` formulation, but with any integer constraints relaxed.
    (3) `mip` solves the full `mip` formulation.
+ `tightening_solver`: Solver used to determine upper and lower bounds for input to nonlinear units.
    Defaults to the same type of solver as the `main_solver`, with a time limit of 20s per solver
    and output suppressed. Used only if the `tightening_algorithm` is `lp` or `mip`.
+ `cache_model`: Defaults to `true`. If `true`, saves model generated. If `false`, does not save model
    generated, but any existing cached model is retained.
+ `solve_if_predicted_in_targeted`: Defaults to `true`. The prediction that `nn` makes for the unperturbed
    `input` can be determined efficiently. If the predicted index is one of the indexes in `target_selection`,
    we can skip the relatively costly process of building the model for the MIP problem since we already have an
    "adversarial example" --- namely, the input itself. We continue build the model and solve the (trivial) MIP
    problem if and only if `solve_if_predicted_in_targeted` is `true`.
"""
function find_adversarial_example(
    nn::NeuralNet,
    input::Array{<:Real},
    target_selection::Union{Integer, Array{<:Integer, 1}},
    invert_target_selection::Bool,
    num_classes::Integer,
    adv_pred_opt::AdversarialPredictionOption,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    pp::PerturbationFamily = UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    rebuild::Bool = false,
    tightening_algorithm::TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = get_default_tightening_solver(main_solver),
    cache_model::Bool = true,
    solve_if_predicted_in_targeted = true,
    adversarial_example_objective::AdversarialExampleObjective = closest
    )::Dict

    total_time = @elapsed begin
        d = Dict()

        # Calculate predicted index
        predicted_output = input |> nn

        predicted_index = predicted_output[1:num_classes] |> get_max_index
        d[:PredictedIndex] = predicted_index

        # If there is a distance threshold, the output must include the distances
        if adv_pred_opt == FindMinDistanceThreshold() || isa(adv_pred_opt, MinDistanceThreshold)
            @assert(length(predicted_output) == num_classes
                + num_classes * (num_classes - 1) / 2)
            d[:NaturalInputDistances] = get_class_distances(predicted_output[num_classes + 1:end], predicted_index, num_classes)
        else
            @assert(length(predicted_output) == num_classes)
        end

        # Set target indexes
        d[:TargetIndexes] = get_target_indexes(target_selection,
                                               num_classes,
                                               invert_target_selection = invert_target_selection)

        if d[:PredictedIndex] in d[:TargetIndexes]
            warn(MIPVerify.LOGGER, "Image is predicted incorrectly, so is already adversarial")
        end

        notice(MIPVerify.LOGGER, "Attempting to find adversarial example. Neural net predicted label is $(predicted_index), target labels are $(d[:TargetIndexes])")

        # Only call solver if predicted index is not found among target indexes.
        if !(d[:PredictedIndex] in d[:TargetIndexes]) || solve_if_predicted_in_targeted
            merge!(
                d,
                get_model(nn, input, pp, tightening_solver, tightening_algorithm, rebuild, cache_model)
            )
            m = d[:Model]

            class_output = d[:Output][1:num_classes]
            dist_output = d[:Output][num_classes + 1:end]

            (maximum_target_var, other_vars) = get_vars_for_max_index(class_output,
                                                                      d[:TargetIndexes],
                                                                      tolerance)

            if adversarial_example_objective == closest
                @constraint(m, other_vars - maximum_target_var .<= -tolerance)
                # Set perturbation objective
                # NOTE (vtjeng): It is important to set the objective immediately before we carry out
                # the solve. Functions like `set_max_indexes` can modify the objective.
                @objective(m, Min, get_norm(norm_order, d[:Perturbation]))
            elseif adversarial_example_objective == worst
                maximum_other_var = maximum_ge(other_vars)
                @objective(m, Max, maximum_target_var - maximum_other_var)
            else
                error("Unknown adversarial_example_objective $adversarial_example_objective")
            end

            setsolver(m, main_solver)

            solve_time = @elapsed begin
                d[:SolveStatus] = solve(m)

                if ((d[:SolveStatus] in [:Optimal, :Suboptimal, :UserObjLimit]) &&
                        (isa(adv_pred_opt, MinDistanceThreshold) ||
                         adv_pred_opt == FindMinDistanceThreshold()))

                    class_output_value = getvalue(class_output)
                    dist_output_value = getvalue(dist_output)

                    # We have to do this manually because sometimes the true class has
                    # has same value as the adversarial predicted class
                    adv_predicted_class = -1
                    max_adv_predicted_val = -Inf
                    for k in d[:TargetIndexes]
                        if class_output_value[k] > max_adv_predicted_val
                            adv_predicted_class = k
                            max_adv_predicted_val = class_output_value[k]
                        end
                    end

                    predicted_class_dists = get_class_distances(dist_output_value, adv_predicted_class, num_classes)

                    notice(MIPVerify.LOGGER, "Outputs: $(class_output_value)")
                    notice(MIPVerify.LOGGER, "Class distances: $(predicted_class_dists)")

                    if isa(adv_pred_opt, MinDistanceThreshold)
                        distance_threshold = adv_pred_opt.threshold
                        if all(predicted_class_dists .>= distance_threshold)
                            warn(MIPVerify.LOGGER, "Solution would *not* be predicted to be adversarial")
                        else
                            warn(MIPVerify.LOGGER, "Solution *would* be predicted to be adversarial")
                            d[:SolveStatus] = verify_distances(nn, input,
                                                d[:TargetIndexes],
                                                adv_predicted_class,
                                                d[:PredictedIndex], num_classes,
                                                distance_threshold, main_solver,
                                                pp, norm_order, tolerance,
                                                rebuild, tightening_algorithm,
                                                tightening_solver, cache_model)
                        end
                    elseif adv_pred_opt == FindMinDistanceThreshold()
                        d[:SolveStatus] = find_min_distance_threshold(nn, input,
                                            d[:TargetIndexes],
                                            adv_predicted_class,
                                            d[:PredictedIndex], num_classes,
                                            main_solver, pp, norm_order,
                                            tolerance, rebuild,
                                            tightening_algorithm,
                                            tightening_solver, cache_model)
                    end
                end
            end
            d[:SolveTime] = try
                getsolvetime(m)
            catch err
                # CBC solver, used for testing, does not implement `getsolvetime`.
                isa(err, MethodError) || rethrow(err)
                solve_time
            end
        end
    end
    d[:TotalTime] = total_time
    return d
end

function find_min_distance_threshold(
    nn::NeuralNet,
    input::Array{<:Real},
    target_indices::Array{<:Integer, 1},
    priority_class_index::Integer,
    predicted_index::Integer,
    num_classes::Integer,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    pp::PerturbationFamily,
    norm_order::Real,
    tolerance::Real,
    rebuild::Bool,
    tightening_algorithm::TighteningAlgorithm,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    cache_model::Bool)

    lower_bound = -2.4
    upper_bound = 4

    undecided = false

    while upper_bound - lower_bound > 0.101
        mid_point = round((upper_bound + lower_bound) / 2, 1)
        solve_status = verify_distances(nn, input, target_indices,
                            priority_class_index, predicted_index, num_classes,
                            mid_point, main_solver, pp, norm_order, tolerance,
                            rebuild, tightening_algorithm, tightening_solver,
                            cache_model)
        if solve_status == :InfeasibleDistance
            upper_bound = mid_point
        elseif solve_status in [:Optimal, :Suboptimal, :UserObjLimit]
            lower_bound = mid_point
        else # The verification failed
            undecided = true
            lower_bound = mid_point
        end
    end

    if undecided
        warn(MIPVerify.LOGGER, "Unable to find true distance threshold, $(upper_bound), $(lower_bound)")
        return string("InfeasibleUndecidedDistance", upper_bound)
    else
        warn(MIPVerify.LOGGER, "Found distance threshold: $(upper_bound)")
        return string("InfeasibleDistance", upper_bound)
    end
end

function verify_distances(
    nn::NeuralNet,
    input::Array{<:Real},
    target_indices::Array{<:Integer, 1},
    priority_class_index::Integer,
    predicted_index::Integer,
    num_classes::Integer,
    distance_threshold::Real,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    pp::PerturbationFamily,
    norm_order::Real,
    tolerance::Real,
    rebuild::Bool,
    tightening_algorithm::TighteningAlgorithm,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    cache_model::Bool)

    warn(MIPVerify.LOGGER, "Starting distance verification, threshold: $(distance_threshold)")
    notice(MIPVerify.LOGGER, "Predicted class: $(predicted_index), target classes: $(target_indices)")

    function test_index(index)
        notice(MIPVerify.LOGGER, "Testing class $(index)")
        d = get_model(nn, input, pp, tightening_solver, tightening_algorithm, rebuild, cache_model)
        m = d[:Model]

        class_output = d[:Output][1:num_classes]
        dist_output = d[:Output][num_classes + 1:end]

        (maximum_target_var, other_vars) = get_vars_for_max_index(class_output,
                                                                  [index],
                                                                  tolerance)
        @constraint(m, other_vars - maximum_target_var .<= -tolerance)

        (pre_indices, post_indices) = get_class_distance_indices(index, num_classes)
        for k in pre_indices
            @constraint(m, dist_output[k] <= -distance_threshold)
        end
        for k in post_indices
            @constraint(m, dist_output[k] >= distance_threshold)
        end

        setsolver(m, main_solver)
        @objective(m, Min, get_norm(norm_order, d[:Perturbation]))
        solve_status = solve(m)

        notice(MIPVerify.LOGGER, "Solve status $(solve_status)")

        if solve_status in [:UserObjLimit, :Optimal, :UserLimit, :Numeric, :Suboptimal, :Cutoff]
            if solve_status in [:Optimal, :Suboptimal, :UserObjLimit]
                notice(MIPVerify.LOGGER, "Outputs: $(getvalue(class_output))")
                notice(MIPVerify.LOGGER, "Class distances: $(get_class_distances(getvalue(dist_output), priority_class_index, num_classes))")
            end
            return solve_status
        end
        return nothing
    end

    status = test_index(priority_class_index)
    if !isnothing(status)
        return status
    end

    for index in target_indices
        if index == priority_class_index
            continue
        end
        status = test_index(index)
        if !isnothing(status)
            return status
        end
    end

    warn(MIPVerify.LOGGER, "Not vulnerable because of distance threshold")
    # If we get here then examples are infeasible *because* of the distance threshold
    return :InfeasibleDistance
end

function get_class_distance_indices(i, num_classes)::Tuple{Array{Integer, 1}, Array{Integer, 1}}
    @assert(1 <= i <= num_classes)

    indices_pre = []
    indices_post = []

    # When j < i
    for j = 1:i-1
        push!(indices_pre, round(Int, (float(num_classes) - 0.5) * (j-1) - ((j-1)^2)/2 + (i - j)))
    end

    # When i < j
    base_index = round(Int, (float(num_classes) - 0.5) * (i-1) - ((i-1)^2)/2)
    for j = i+1:num_classes
        push!(indices_post, base_index + j - i)
    end

    return (indices_pre, indices_post)
end

function get_class_distances(xs::Array{<:Real, 1}, i, num_classes)::Array{<:Real, 1}
    @assert(length(xs) == num_classes * (num_classes - 1) / 2)
    @assert(1 <= i <= num_classes)

    subset = Real[]

    # When j < i
    for j = 1:i-1
        push!(subset, -1 * xs[round(Int, (float(num_classes) - 0.5) * (j-1) - ((j-1)^2)/2 + (i - j))])
    end

    # When i < j
    base_index = round(Int, (float(num_classes) - 0.5) * (i-1) - ((i-1)^2)/2)
    for j = i+1:num_classes
        push!(subset, xs[base_index + j - i])
    end

    return subset
end

function get_label(y::Array{<:Real, 1}, test_index::Integer)::Int
    return y[test_index]
end

function get_image(x::Array{T, 4}, test_index::Integer)::Array{T, 4} where {T<:Real}
    return x[test_index:test_index, :, :, :]
end

"""
$(SIGNATURES)

Returns the fraction of items the neural network correctly classifies of the first
`num_samples` of the provided `dataset`. If there are fewer than
`num_samples` items, we use all of the available samples.

# Named Arguments:
+ `nn::NeuralNet`: The parameters of the neural network.
+ `dataset::LabelledDataset`:
+ `num_samples::Integer`: Number of samples to use.
"""
function frac_correct(
    nn::NeuralNet, 
    dataset::LabelledDataset, 
    num_samples::Integer)::Real

    num_correct = 0.0
    num_samples = min(num_samples, MIPVerify.num_samples(dataset))
    @showprogress 1 "Computing fraction correct..." for sample_index in 1:num_samples
        x0 = get_image(dataset.images, sample_index)
        actual_label = get_label(dataset.labels, sample_index)
        predicted_label = (x0 |> nn |> get_max_index) - 1
        if actual_label == predicted_label
            num_correct += 1
        end
    end
    return num_correct / num_samples
end


function get_norm(
    norm_order::Real,
    v::Array{<:Real})
    if norm_order == 1
        return sum(abs.(v))
    elseif norm_order == 2
        return sqrt(sum(v.*v))
    elseif norm_order == Inf
        return Base.maximum(Iterators.flatten(abs.(v)))
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

function get_norm(
    norm_order::Real,
    v::Array{<:JuMPLinearType, N}) where {N}
    if norm_order == 1
        abs_v = abs_ge.(v)
        return sum(abs_v)
    elseif norm_order == 2
        return sum(v.*v)
    elseif norm_order == Inf
        return MIPVerify.maximum_ge(flatten(abs_ge.(v), N:-1:1))
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

function isnothing(x)::Bool
    x === nothing
end

include("batch_processing_helpers.jl")

end
