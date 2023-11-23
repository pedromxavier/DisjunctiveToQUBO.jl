using QUBOTools
using QUBODrivers
using LinearAlgebra

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
    
    print(
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

    print(
        """
        ⋄ Sampling Summary

        ⋄ Number of samples: $(ns)

        ⋄ Best energy: $(λ0)

        ⋄ Time-to-target (sec): $(ts)
        """
    )
end
