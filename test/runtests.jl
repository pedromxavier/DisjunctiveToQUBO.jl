using Test
using HiGHS
using ToQUBO
using QUBOTools
using QUBODrivers
using DisjunctiveToQUBO.DP
using DisjunctiveToQUBO

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

    @info "Raw QUBO model:"

    println(qubo_model)

    coef_range(qubo_model)
end

bigm()
hull()
raw()
