using Test
using HiGHS
using ToQUBO
import ToQUBO
using QUBOTools
using QUBODrivers
using DisjunctiveToQUBO.DP
using DisjunctiveToQUBO

const MOI = ToQUBO.MOI
const VI = MOI.VariableIndex
const PBO = ToQUBO.PBO

function MOI.supports_constraint(
    ::ToQUBO.Virtual.Model,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{<:MOI.Indicator{A,S}},
) where {A,S}
    return true
end

function MOI.supports_constraint(
    ::ToQUBO.PreQUBOModel,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{<:MOI.Indicator{A,S}},
) where {A,S}
    return true
end

function ToQUBO.Compiler.constraint(model::ToQUBO.Virtual.Model{T}, x::MOI.VectorOfVariables, ::MOI.SOS1{T}, ::QUBOTools.AbstractArchitecture) where {T}
    # Special Ordered Set of Type 1: ∑ x ≤ min x
    g = PBO.PBF{VI,T}()

    for xi in x.variables
        vi = model.source[xi]

        if ToQUBO.Virtual.encoding(vi) isa ToQUBO.Encoding.Mirror
            for (ωi, _) in ToQUBO.Virtual.expansion(vi)
                g[ωi] = one(T)
            end
        elseif ToQUBO.Virtual.encoding(vi) isa ToQUBO.Encoding.SetVariableEncodingMethod
            flag = false
            
            ξ = ToQUBO.Virtual.expansion(vi)
            a = ξ[nothing]

            for (ωi, ci) in ξ
                isempty(ωi) && continue

                γi = ci + a

                if iszero(γi)
                    flag = true

                    # 1 - y_i
                    g[ωi] = -one(T)

                    g[nothing] += one(T)
                end
            end

            if !flag
                error("Variable '$vi' is always non-zero")
            end
        elseif ToQUBO.Virtual.encoding(vi) isa ToQUBO.Encoding.IntervalVariableEncodingMethod
            # Indicator variable
            u = ToQUBO.Encoding.encode!(model, nothing, ToQUBO.Encoding.Mirror{T}())

            error("Currently, ToQUBO only supports SOS1 on binary variables or arbitrary-set-encoded")
        end
    end

    # Slack variable
    z = ToQUBO.Encoding.encode!(model, nothing, ToQUBO.Encoding.Mirror{T}())

    for (ω, c) in ToQUBO.Virtual.expansion(z)
        g[ω] += c
    end

    g[nothing] += -one(T)

    return g^2
end



function coef_range(m)
    l = last.(QUBOTools.linear_terms(m))
    q = last.(QUBOTools.quadratic_terms(m))

    @info """
        Linear coefficient range: $(isempty(l) || extrema(l))
        Quadratic coefficient range: $(isempty(q) || extrema(q))
        """
end

function bigm()
    model = GDPModel(ToQUBO.Optimizer)

    @variable(model, 0 ≤ x[1:2] ≤ 20)
    @variable(model, Y[1:2], Logical)
    @constraint(model, [i = 1:2], [2,5][i] ≤ x[i] ≤ [6,9][i], Disjunct(Y[1]))
    @constraint(model, [i = 1:2], [8,10][i] ≤ x[i] ≤ [11,15][i], Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, sum(x))

    optimize!(model, method = BigM(100, true)) #specify M value and disable M-tightening

    qubo_model = QUBOTools.Model(unsafe_backend(model).target_model)

    @info "Big-M reformulated QUBO model:"

    println(qubo_model)

    coef_range(qubo_model)
end

function hull()
    model = GDPModel(ToQUBO.Optimizer)

    @variable(model, 0 ≤ x[1:2] ≤ 20)
    @variable(model, Y[1:2], Logical)
    @constraint(model, [i = 1:2], [2,5][i] ≤ x[i] ≤ [6,9][i], Disjunct(Y[1]))
    @constraint(model, [i = 1:2], [8,10][i] ≤ x[i] ≤ [11,15][i], Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, sum(x))

    optimize!(model, method = Hull())

    qubo_model = QUBOTools.Model(unsafe_backend(model).target_model)

    @info "Hull reformulated QUBO model:"

    println(qubo_model)

    coef_range(qubo_model)
end

function ind()
    model = GDPModel(ToQUBO.Optimizer)

    @variable(model, 0 ≤ x[1:2] ≤ 20)
    @variable(model, Y[1:2], Logical)
    @constraint(model, [i = 1:2], [2,5][i] ≤ x[i] ≤ [6,9][i], Disjunct(Y[1]))
    @constraint(model, [i = 1:2], [8,10][i] ≤ x[i] ≤ [11,15][i], Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, sum(x))

    optimize!(model; method = Indicator())

    qubo_model = QUBOTools.Model(unsafe_backend(model).target_model)

    @info "Indicator reformulated QUBO model:"

    println(qubo_model)

    coef_range(qubo_model)
end

function raw()
    model = GDPModel(ToQUBO.Optimizer)

    @variable(model, 0 ≤ x[1:2] ≤ 20)
    @variable(model, Y[1:2], Logical)
    @constraint(model, [i = 1:2], [2,5][i] ≤ x[i] ≤ [6,9][i], Disjunct(Y[1]))
    @constraint(model, [i = 1:2], [8,10][i] ≤ x[i] ≤ [11,15][i], Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, sum(x))

    optimize!(model; ignore_optimize_hook = true)

    qubo_model = QUBOTools.Model(unsafe_backend(model).target_model)

    @info "------------------------------\nRaw input model:"

    println(unsafe_backend(model).source_model)

    @info "------------------------------\nRaw QUBO model:"

    println(qubo_model)

    coef_range(qubo_model)
end

bigm()
hull()
ind()
raw()
