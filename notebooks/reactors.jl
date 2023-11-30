using Revise
using JuMP
import GR
using Plots
using LinearAlgebra
using DisjunctiveProgramming

const diamond = raw"$\diamond$"

includet("paper.jl")

# Set parameters
cx = [1.5, 1]
cy = [6.1, 5.9]
α  = [0.75, 0.8]
d  = 3 # not specified in the problem, I think it is 3

function solve_gdp_reactors(config!::Function, optimizer=HiGHS.Optimizer; method=Indicator())
    model = GDPModel(optimizer)

    @variable(model, 0 <= x[1:2] <= 5)
    @variable(model, Y[1:2], Logical)

    y = binary_variable.(Y)

    @objective(model, Min, x' * diagm(cx) * x + cy'y)

    @constraint(model, α'x ≥ d)

    @constraint(model, x[2] ≤ 0, Disjunct(Y[1]))
    @constraint(model, x[1] ≤ 0, Disjunct(Y[2]))

    disjunction(model, Y, exactly1=true)

    config!(model)

    optimize!(model; gdp_method=method)

    return model
end

solve_gdp_reactors(optimizer=HiGHS.Optimizer; method=Indicator()) = solve_gdp_reactors(identity, optimizer; method)

function plot_reactor_base()
    plt = plot_base()

    return plot!(
        plt;
        title          = raw"$ \min~z = \mathbf{x}' \textrm{diag}(\mathbf{c_{x}})\, \mathbf{x} + \mathbf{c_{y}}'\mathbf{y} $",
        colorbar_title = raw"$z$",
        xlims          = ( -1,  6),
        ylims          = ( -1,  6),
        clims          = ( 20, 45),
    )
end

function plot_reactor_feasible(; ns::Integer=1_000, color=:bluesreds)
    plt = plot_reactor_base()

    plot!(plt; plot_title="Feasible Region")

    plot_reactor_feasible!(plt; ns, color)

    plot!(plt; legend_columns = -1)

    return plt
end

function plot_reactor_feasible!(plt; ns::Integer=1_000, na::Integer=40, la::Real=0.15, color=:bluesreds)
    x1 = filter(x -> α[1] * x ≥ d, range(0, 5; length=ns))
    x2 = filter(x -> α[2] * x ≥ d, range(0, 5; length=ns))
    xs = range(-1, 6; length=ns)

    plot!(
        plt,
        xs, (x) -> (d - α[1] * x) / α[2];
        label     = raw"$\alpha'\mathbf{x} \geq d$",
        color     = :gray,
        linestyle = :dash,
        # z_order   = :back,
    )

    # arrows
    xa = range(-1, 6; length=na)
    ya = map((x) -> (d - α[1] * x) / α[2], xa)
    la = sqrt(α[1]^2 + α[2]^2) / la
    ua = fill(α[1] / la, na)
    va = fill(α[2] / la, na)

    GR.setarrowsize(0.5)

    quiver!(
        plt,
        xa - ua, ya - va;
        color   = :gray,
        quiver  = (ua, va),
        label   = nothing,
        arrow   = Plots.arrow(:closed),
        # z_order = :back,
    )

    z1 = collect(map(x -> cx[1] * x ^ 2 + cy[2], x1))
    z2 = collect(map(x -> cx[2] * x ^ 2 + cy[1], x2))

    plot!(
        plt,
        x1, zeros(length(x1));
        linez     = z1,
        color     = color,
        label     = nothing, # raw"$y_{2} = 1$",
        linewidth = 5,
        # z_order   = :back,
        # colorbar_entry = false,
    )

    plot!(
        plt,
        zeros(length(x2)), x2;
        linez     = z2,
        color     = color,
        label     = nothing, # raw"$y_{1} = 1$",
        linewidth = 5,
        # z_order   = :back,
        # colorbar_entry = false,
    )

    return plt
end

function plot_reactor_optimal(x⃰; ns::Integer = 1_000, color=:bluesreds)
    plt = plot_reactor_feasible(; ns, color)

    plot_optimal_solution!(plt, x⃰)

    plot!(plt; plot_title="Feasible Region $(diamond) Optimal Solution", legend_columns = -1)

    return plt
end


function plot_reactor_bigm(
    model::JuMP.Model,
    x⃰::Vector{T},
    M::Number;
    nr::Integer=result_count(model),
    ns::Integer=1_000,
    color=:bluesreds,
    plot_title="", feasible::Function = <(0),
) where {T}
    x = reverse!([value.(model[:x]; result=i) for i = 1:nr])
    z = reverse!([objective_value(model; result=i) for i = 1:nr])
    r = reverse!([reads(model; result=i) for i = 1:nr])

    return plot_reactor_bigm(x, z, r, x⃰, M; ns, color, plot_title, feasible)
end

function plot_reactor_bigm(x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; ns::Integer=1_000, color=:bluesreds, plot_title="", feasible::Function = <(0)) where {T}
    plt = plot_reactor_feasible(; ns, color)
    
    return plot_reactor_bigm!(plt, x, z, r, x⃰, M; plot_title, feasible)
end

function plot_reactor_bigm!(plt, x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; plot_title="", feasible::Function = <(0)) where {T}
    plot_reactor_bigm_relaxation!(plt, M)
    
    plot_solutions!(plt, x, z, r, x⃰; feasible)
    
    plot!(plt; plot_title, legend_columns = -1)

    return plt
end

function plot_reactor_bigm_relaxation!(plt, M::Number = 5.0; ns::Integer=1_000)
    x1 = x2 = range(0, 5; length=ns)

    envelope(x1, x2) = (0 <= x1 <= M) &&
                       (0 <= x2 <= M) &&
                       (0 <= x1 <= 5) &&
                       (0 <= x2 <= 5)

    shading(x1, x2) = ifelse(envelope(x1, x2), 1.0, NaN)

    heatmap!(
        plt, x1, x2, shading;
        color          = :red,
        alpha          = 0.2,
        colorbar_entry = false,
    )

    return plt
end

function plot_reactor_hull_base(nr::Integer)
    plt = plot_reactor_base()

    return plot!(
        plt;
        plot_title     = "Hull Feasible Region, samples = $(nr)",
    )
end

function plot_reactor_hull_shade!(plt; ns::Integer=1_000, color=:bluesreds)
    x1 = x2 = range(0, 5; length=ns)

    envelope(x1, x2) = (0 ≤ x2 + x1 ≤ 5) && (0 ≤ x1 ≤ 5) && (0 ≤ x2 ≤ 5) 
    shading(x1, x2)  = ifelse(envelope(x1, x2), 1.0, NaN)

    heatmap!(
        plt, x1, x2, shading;
        color          = :red,
        alpha          = 0.2,
        z_order        = :back,
        colorbar_entry = false,
    )

    return plot_reactor_feasible!(plt; ns, color)
end

function plot_reactor_hull(model::JuMP.Model, x⃰::Vector; nr::Integer=result_count(model), ns::Integer=1_000, color=:bluesreds, plot_title="", feasible = <(0))
    plt = plot_reactor_hull_base(nr)

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z  = reverse!([objective_value(model; result=i) for i = 1:nr])
    r  = reverse!([reads(model; result=i) for i = 1:nr])
    x  = collect.(zip(x1, x2))

    plot_reactor_hull_shade!(plt; ns, color)

    plot_solutions!(plt, x, z, r, x⃰; color, feasible)

    plot!(plt; plot_title, legend_columns = -1)

    return plt
end

function plot_reactor_indicator_base(nr::Integer)
    plt = plot_reactor_base()

    return plot!(plt;
        plot_title = "Indicator Feasible Region, samples = $(nr)",
    )
end

function plot_reactor_indicator(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer=1_000, color=:bluesreds, plot_title="", feasible = <(0))
    plt = plot_reactor_indicator_base(nr)

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z  = reverse!([objective_value(model; result=i) for i = 1:nr])
    r  = reverse!([reads(model; result=i) for i = 1:nr])
    x  = collect.(zip(x1, x2))

    plot_reactor_feasible!(plt; ns)

    plot_solutions!(plt, x, z, r, x⃰; color, feasible)

    plot!(plt; plot_title, legend_columns = -1)

    return plt
end

function solve_indint_reactors(optimizer = DWave.Neal.Optimizer)
    return solve_indint_reactors(identity, optimizer)
end

function solve_indint_reactors(config!::Function, optimizer = DWave.Neal.Optimizer)
    model = Model(() -> ToQUBO.Optimizer(optimizer))

    @variable(model, 0 <= x[1:2] <= 5)
    @variable(model, y[1:2], Bin)

    @objective(model, Min, x' * diagm(cx) * x + cy'y)
    
    @constraint(model, α'x ≥ d)
    
    @constraint(model, sum(y) == 1)
    # @constraint(model, y[1] => {x[2] ≤ 0.0})
    # @constraint(model, y[2] => {x[1] ≤ 0.0})

    let virtual_model = unsafe_backend(model)::ToQUBO.Optimizer
        virtual_model.compiler_settings[:setup_callback] = m -> begin
            y1 = only(ToQUBO.Encoding.encode!(m, y[1].index, ToQUBO.Encoding.Mirror()).y)
            y2 = only(ToQUBO.Encoding.encode!(m, y[2].index, ToQUBO.Encoding.Mirror()).y)

            let x1 = x[1].index
                n  = MOI.get(m, ToQUBO.Attributes.VariableEncodingBits(), x1)
                z1, ξ, _ = ToQUBO.Encoding.encode(ToQUBO.Encoding.Unary(), (0.0, 5.0), n) do (nv::Union{Integer,Nothing} = nothing)
                    if isnothing(nv)
                        return MOI.add_variable(m.target_model)
                    else
                        return MOI.add_variables(m.target_model, nv)
                    end
                end

                ξ1 = ToQUBO.PBO.PBF{ToQUBO.VI,Float64}(y2) * ξ
                v1 = ToQUBO.Virtual.Variable{Float64}(
                    DisjunctEncoding(), # new encoding method
                    x1,
                    z1,
                    ξ1,
                    nothing,
                )

                ToQUBO.Encoding.encode!(m, v1)
            end

            let x2 = x[2].index
                n  = MOI.get(m, ToQUBO.Attributes.VariableEncodingBits(), x2)
                z2, ξ, _ = ToQUBO.Encoding.encode(ToQUBO.Encoding.Unary(), (0.0, 5.0), n) do (nv::Union{Integer,Nothing} = nothing)
                    if isnothing(nv)
                        return MOI.add_variable(m.target_model)
                    else
                        return MOI.add_variables(m.target_model, nv)
                    end
                end

                ξ2 = ToQUBO.PBO.PBF{ToQUBO.VI,Float64}(y1) * ξ
                v2 = ToQUBO.Virtual.Variable{Float64}(
                    DisjunctEncoding(), # new encoding method
                    x2,
                    z2,
                    ξ2,
                    nothing,
                )

                ToQUBO.Encoding.encode!(m, v2)
            end

            MOI.set(m, ToQUBO.Attributes.Quadratize(), true)
        end
    end

    config!(model)

    optimize!(model)

    return model
end


function solve_indintcons_reactors(optimizer = DWave.Neal.Optimizer)
    return solve_indintcons_reactors(identity, optimizer)
end

function solve_indintcons_reactors(config!::Function, optimizer = DWave.Neal.Optimizer)
    model = Model(() -> ToQUBO.Optimizer(optimizer))

    @variable(model, 0 <= x[1:2] <= 5)
    @variable(model, y[1:2], Bin)

    @objective(model, Min, x' * diagm(cx) * x + cy'y)
    
    # @constraint(model, α'x ≥ d)
    
    @constraint(model, sum(y) == 1)
    # @constraint(model, y[1] => {x[2] ≤ 0.0})
    # @constraint(model, y[2] => {x[1] ≤ 0.0})

    let virtual_model = unsafe_backend(model)::ToQUBO.Optimizer
        virtual_model.compiler_settings[:setup_callback] = m -> begin
            y1 = only(ToQUBO.Encoding.encode!(m, y[1].index, ToQUBO.Encoding.Mirror()).y)
            y2 = only(ToQUBO.Encoding.encode!(m, y[2].index, ToQUBO.Encoding.Mirror()).y)

            let x1 = x[1].index
                n  = MOI.get(m, ToQUBO.Attributes.VariableEncodingBits(), x1)

                z1, ξ, _ = ToQUBO.Encoding.encode(ToQUBO.Encoding.Unary(), (d / α[1], 5.0), n) do (nv::Union{Integer,Nothing} = nothing)
                    if isnothing(nv)
                        return MOI.add_variable(m.target_model)
                    else
                        return MOI.add_variables(m.target_model, nv)
                    end
                end

                ξ1 = ToQUBO.PBO.PBF{ToQUBO.VI,Float64}(y2) * ξ
                v1 = ToQUBO.Virtual.Variable{Float64}(
                    DisjunctEncoding(), # new encoding method
                    x1,
                    z1,
                    ξ1,
                    nothing,
                )

                ToQUBO.Encoding.encode!(m, v1)
            end

            let x2 = x[2].index
                n  = MOI.get(m, ToQUBO.Attributes.VariableEncodingBits(), x2)

                z2, ξ, _ = ToQUBO.Encoding.encode(ToQUBO.Encoding.Unary(), (d / α[2], 5.0), n) do (nv::Union{Integer,Nothing} = nothing)
                    if isnothing(nv)
                        return MOI.add_variable(m.target_model)
                    else
                        return MOI.add_variables(m.target_model, nv)
                    end
                end

                ξ2 = ToQUBO.PBO.PBF{ToQUBO.VI,Float64}(y1) * ξ
                v2 = ToQUBO.Virtual.Variable{Float64}(
                    DisjunctEncoding(), # new encoding method
                    x2,
                    z2,
                    ξ2,
                    nothing,
                )

                ToQUBO.Encoding.encode!(m, v2)
            end

            MOI.set(m, ToQUBO.Attributes.Quadratize(), true)
        end
    end

    config!(model)

    optimize!(model)

    return model
end