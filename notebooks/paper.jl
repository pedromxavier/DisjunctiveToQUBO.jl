using Revise
using JuMP
using DWave
using ToQUBO
using QUBOTools
using QUBODrivers
using LinearAlgebra
using PythonCall
using Plots

const diamond = raw"$\diamond$"

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
    tf::Float64
end

function Base.show(io::IO, ss::SamplingSummary)
    println(
        io,
        """
        ⋄ Sampling Summary

        ⋄ Number of samples: $(ss.ns)

        ⋄ Best energy: $(ss.λ0)

        ⋄ Time-to-target (sec): $(ss.ts)

        ⋄ Time-to-feasible (sec): $(ss.tf)
        """
    )
end

function sampling_summary(model, λ, μ = 0.0)
    # Retrieve Virtual Model
    vm = unsafe_backend(model)::ToQUBO.Optimizer

    vm.optimizer::QUBODrivers.AbstractSampler

    # Retrieve QUBO Optimizer
    qo = QUBOTools.backend(vm.optimizer)

    # Retrieve Solution
    ss = QUBOTools.solution(qo)
    ns = length(ss)

    ts = QUBOTools.ttt(ss, λ) |> abs
    tf = QUBOTools.ttt(ss, μ) |> abs

    λ0 = if ns > 0
        ss[1].value
    else
        NaN
    end

    return SamplingSummary(ns, λ0, ts, tf)
end

function table_summary(data, λ, μ)
    header = join(
        [
            "Reformulation Method",
            raw"$\log_{10}\Delta$",
            raw"$n_{\textrm{vars}}$",
            raw"$\textrm{TTT}_{\textrm{SA}}$",
            raw"$\textrm{TTF}_{\textrm{SA}}$",
            raw"$n_{\textrm{qubits}}$",
            raw"$\textrm{TTT}_{\textrm{QA}}$",
            raw"$\textrm{TTF}_{\textrm{QA}}$",
        ],
        " & ",
    ) * raw" \\\\"

    rows = []

    for (method, (sa_model, qa_model)) in data
        cs    = compilation_summary(qa_model)
        sa_ss = sampling_summary(sa_model, λ, μ)
        qa_ss = sampling_summary(qa_model, λ, μ)

        Δ = trunc(Int, log10(max(abs(cs.l), abs(cs.u))))

        φ = (x) -> x isa Float64 ? (isinf(x) ? raw"$\infty$" : string(round(x; digits=2))) : string(x)

        push!(
            rows,
            [
                method,
                φ(Δ),
                φ(cs.n),
                φ(sa_ss.ts),
                φ(sa_ss.tf),
                φ(cs.qb),
                φ(qa_ss.ts),
                φ(qa_ss.tf),
            ],
        )
    end

    l = [maximum([length(rows[j][i]) for j = eachindex(rows)]) for i = 1:8]

    for i = eachindex(rows)
        for j = eachindex(rows[i])
            rows[i][j] = rpad(rows[i][j], l[j])
        end

        rows[i] = "  " * join(rows[i], " & ") * raw" \\\\"
    end

    table = join(
        [
            raw"\hline",
            header,
            raw"\hline",
            rows...,
            raw"\hline",
        ],
        "\n",
    )

    return """
    \\begin{tabular}{|l|ccccccc|}
    $(table)
    \\end{tabular}
    """
end

includet("paper_plots.jl")
