using JuMP
using ToQUBO
using QUBOTools
using QUBODrivers
using LinearAlgebra
using Plots

export NumberOfReads, reads, compilation_summary, sampling_summary, squares

# Make plots look professional
Plots.default(;
    fontfamily = "Computer Modern",
    plot_titlefontsize  = 16,
    titlefontsize       = 14,
    guidefontsize       = 12,
    legendfontsize      = 10,
    tickfontsize        = 10,
)

const NumberOfReads = QUBOTools.__moi_num_reads()

function reads(model::JuMP.Model; result::Integer = 1)
    vm = unsafe_backend(model)::ToQUBO.Optimizer
    qo = vm.optimizer::QUBODrivers.AbstractSampler

    return QUBOTools.reads(qo, result)
end

function compilation_summary(model)
    # Retrieve Virtual Model
    vm = unsafe_backend(model)::ToQUBO.Optimizer

    ev = 0
    sv = 0
    qv = 0

    for v in vm.variables
        let src = ToQUBO.Virtual.source(v)
            if isnothing(src)
                qv += length(ToQUBO.Virtual.target(v))
            elseif src isa MOI.VariableIndex
                ev += length(ToQUBO.Virtual.target(v))
            elseif src isa MOI.ConstraintIndex
                sv += length(ToQUBO.Virtual.target(v))
            else
                error()
            end
        end
    end

    # Retrieve QUBO Model
    qm = QUBOTools.Model(vm.target_model)

    ld = QUBOTools.linear_density(qm)
    qd = QUBOTools.quadratic_density(qm)

    solver_name = if !isnothing(vm.optimizer)
        MOI.get(vm.optimizer, MOI.SolverName())
    else
        "None (Compilation Mode)"
    end

    n, L, Q, α, β = QUBOTools.qubo(qm, :dense)

    QM = Q + diagm(L)

    l, u = extrema(QM)

    t = MOI.get(vm, ToQUBO.Attributes.CompilationTime())
    
    println(
        """
        ⋄ Compilation Summary

        ⋄ Solver: $(solver_name)

        ⋄ Number of variables: $(n)
          Encoding ………………… $(ev)
          Slack ………………………… $(sv)
          Quadratization … $(qv)

        ⋄ Density
          Linear ……………………… $(ld)
          Quadratic ……………… $(qd)

        ⋄ Coefficient Range
          Lower Bound ………… $(l)
          Upper Bound ………… $(u)

        ⋄ Constant Offset: $(β)

        ⋄ Work counters
          Compiltaion Time (sec) : $(t)
        """
    )
end

function sampling_summary(model, λ)
    # Retrieve Virtual Model
    vm = unsafe_backend(model)::ToQUBO.Optimizer

    vm.optimizer::QUBODrivers.AbstractSampler

    # Retrieve QUBO Optimizer
    qo = QUBOTools.backend(vm.optimizer)

    # Retrieve Solution
    ss = QUBOTools.solution(qo)
    ns = length(ss)

    ts = QUBOTools.ttt(ss, λ)

    λ0 = if ns > 0
        ss[1].value
    else
        NaN
    end

    println(
        """
        ⋄ Sampling Summary

        ⋄ Number of samples: $(ns)

        ⋄ Best energy: $(λ0)

        ⋄ Time-to-target (sec): $(ts)
        """
    )
end

function plot_optimal_solution!(plt, x⃰; s = 1.0)
    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color      = :white,
        marker     = :star8,
        markersize = 10,
        label="Optimal Solution",
    )

    return plt
end

function plot_best_sample!(plt, x::Vector{Vector{T}}) where {T}
    scatter!(
        plt,
        [x[end][1]],
        [x[end][2]];
        color      = :violet,
        marker     = :rect,
        label      = "Best Sample",
        markersize = 10,
    )

    return plt
end

function plot_solutions!(plt, x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}; nr::Integer=length(x), color=:balance) where {T}
    scatter!(
        plt,
        [x[i][1] for i = 1:nr],
        [x[i][2] for i = 1:nr];
        zcolor         = z,
        color          = color,
        marker         = :circle,
        markersize     = 2r,
        legend_columns = 2,
        label          = "Samples",
    )

    plot_best_sample!(plt, x)

    return plot_optimal_solution!(plt, x⃰)
end
