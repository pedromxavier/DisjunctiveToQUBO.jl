using Revise
using JuMP
using DWave
using ToQUBO
using QUBOTools
using QUBODrivers
using LinearAlgebra
using PythonCall
using Plots

# export NumberOfReads, reads, compilation_summary, sampling_summary, squares
struct DisjunctEncoding{T} <: ToQUBO.Encoding.VariableEncodingMethod end

DisjunctEncoding() = DisjunctEncoding{Float64}()

ToQUBO.PBO.varshow(v::ToQUBO.VI) = ToQUBO.PBO.varshow(v.value)

# Make plots look professional
Plots.default(;
    fontfamily = "Computer Modern",
    plot_titlefontsize     = 16,
    titlefontsize          = 14,
    colorbar_titlefontsize = 14,
    guidefontsize          = 12,
    legendfontsize         = 10,
    tickfontsize           = 10,
)

const NumberOfReads = QUBOTools.__moi_num_reads()

function reads(model::JuMP.Model; result::Integer = 1)
    vm = unsafe_backend(model)::ToQUBO.Optimizer
    qo = vm.optimizer::QUBODrivers.AbstractSampler

    return QUBOTools.reads(qo, result)
end

struct CompilationSummary
    solver_name::String
    n::Int
    ev::Int
    sv::Int
    qv::Int
    ld::Float64
    qd::Float64
    l::Float64
    u::Float64
    β::Float64
    t::Float64
    qb::Union{Int,Nothing}
end

function Base.show(io::IO, cs::CompilationSummary)
    println(
        io,
        """
        ⋄ Compilation Summary

        ⋄ Solver: $(cs.solver_name)

        ⋄ Number of variables: $(cs.n)
          Encoding ………………… $(cs.ev)
          Slack ………………………… $(cs.sv)
          Quadratization … $(cs.qv)

        ⋄ Number of qubits: $(cs.qb)

        ⋄ Density
          Linear ……………………… $(cs.ld)
          Quadratic ……………… $(cs.qd)

        ⋄ Coefficient Range
          Lower Bound ………… $(cs.l)
          Upper Bound ………… $(cs.u)

        ⋄ Constant Offset: $(cs.β)

        ⋄ Work counters
          Compilation Time (sec) : $(cs.t)
        """
    )
end

function qubits(optimizer::DWave.Optimizer)::Integer
    e = optimizer.model.solution.metadata["dwave_info"]["embedding_context"]["embedding"]

    return length(reduce(union!, Set{Int}.(values(e))))
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

    qb = if vm.optimizer isa DWave.Optimizer
        qubits(vm.optimizer)
    else
        nothing
    end
    
    return CompilationSummary(
        solver_name,
        n,
        ev,
        sv,
        qv,
        ld,
        qd,
        l,
        u,
        β,
        t,
        qb,
    )
end

struct SamplingSummary
    ns::Int
    λ0::Float64
    ts::Float64
end

function Base.show(io::IO, ss::SamplingSummary)
    println(
        io,
        """
        ⋄ Sampling Summary

        ⋄ Number of samples: $(ss.ns)

        ⋄ Best energy: $(ss.λ0)

        ⋄ Time-to-target (sec): $(ss.ts)
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

    return SamplingSummary(ns, λ0, ts)
end

includet("paper_plots.jl")
