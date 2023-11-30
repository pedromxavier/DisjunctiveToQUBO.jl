include("paper.jl")

using JuMP
using Plots
using DisjunctiveProgramming

function solve_gdp_squares(config!::Function, optimizer=HiGHS.Optimizer; method=Indicator())
    model = GDPModel(optimizer)

    @variable(model, -2 <= x[1:2] <= 2)
    @variable(model, Y[1:2], Logical)
    y = binary_variable.(Y)

    @objective(model, Min, x[1] - x[2] + y[1] + 5y[2])

    @constraint(model, -2 .≤ x .≤ -1, Disjunct.([Y[1], Y[1]]))
    @constraint(model, 1 .≤ x .≤ 2, Disjunct.([Y[2], Y[2]]))

    disjunction(model, Y, exactly1=true)

    config!(model)

    optimize!(model; gdp_method=method)

    return model
end

solve_gdp_squares(optimizer=HiGHS.Optimizer; method=Indicator()) = solve_gdp_squares(identity, optimizer; method)

function plot_square_base()
    plt = plot_base()

    return plot!(
        plt;
        title          = raw"$ \min~z = x_{1} - x_{2} + y_{1} + 5y_{2} $",
        colorbar_title = raw"$z$",
        xlims          = (-3, 3),
        ylims          = (-3, 3),
        clims          = ( 0, 6),
        legend_columns = -1,
    )
end

function plot_square_feasible(; ns::Integer=1_000, color=:bluesreds)
    plt = plot_square_base()

    return plot_square_feasible!(plt; ns, color)
end

function plot_square_feasible!(plt; ns::Integer=1_000, color=:bluesreds)
    plot!(plt; plot_title="Feasible Region")

    x1 = x2 = range(-2, 2; length=ns)

    # y1 = 1
    objective1(x1, x2) = x1 - x2 + 1
    feasible1(x1, x2)  = -2 ≤ x1 ≤ -1 && -2 ≤ x2 ≤ -1
    coloring1(x1, x2)  = ifelse(feasible1(x1, x2), objective1(x1, x2), NaN)

    # y2 = 1
    objective2(x1, x2) = x1 - x2 + 5
    feasible2(x1, x2)  = 1 ≤ x1 ≤ 2 && 1 ≤ x2 ≤ 2
    coloring2(x1, x2)  = ifelse(feasible2(x1, x2), objective2(x1, x2), NaN)

    heatmap!(
        plt,
        x1, x2,
        coloring1;
        color=color,
    )

    heatmap!(
        plt,
        x1, x2,
        coloring2;
        color=color,
    )

    return plt
end

function plot_square_bigm(
    model::JuMP.Model,
    x⃰::Vector{T},
    M::Number;
    nr::Integer=result_count(model),
    ns::Integer=1_000,
    color=:bluesreds,
    plot_title="",
    feasible::Function=<(0),
) where {T}
    x = reverse!([value.(model[:x]; result=i) for i = 1:nr])
    z = reverse!([objective_value(model; result=i) for i = 1:nr])
    r = reverse!([reads(model; result=i) for i = 1:nr])

    plt = plot_square_bigm(x, z, r, x⃰, M; ns, color, feasible)

    plot!(plt; plot_title)

    return plt
end

function plot_square_bigm(x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; ns::Integer=1_000, color=:bluesreds, feasible::Function=<(0)) where {T}
    plt = plot_square_feasible(; ns, color)

    return plot_square_bigm!(plt, x, z, r, x⃰, M; feasible)
end

function plot_square_bigm!(plt, x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; feasible::Function=<(0)) where {T}
    plot_square_bigm_relaxation!(plt, M)

    return plot_solutions!(plt, x, z, r, x⃰; feasible)
end

function plot_square_bigm_relaxation!(plt, M::Number; ns::Integer=1_000)
    x1 = x2 = range(-2, 2; length=ns)

    envelope(x1, x2, M=M) = (-2 - M <= x1 <= -1 + M) &&
                            (-2 - M <= x2 <= -1 + M) &&
                            (1 - M <= x1 <= 2 + M) &&
                            (1 - M <= x2 <= 2 + M)

    shading(x1, x2) = ifelse(envelope(x1, x2), 1.0, NaN)

    heatmap!(
        plt, x1, x2, shading;
        color=:red,
        alpha=0.2,
        colorbar_entry=false,
    )

    return plt
end

function plot_square_hull_base()
    plt = plot_base()

    return plot!(
        plt;
        title        = raw"$ \min~z = x_{1} - x_{2} + y_{1} + 5y_{2} $",
        colorbar_title = raw"$z$",
        xlims        = (-3, 3),
        ylims        = (-3, 3),
        clims        = ( 0, 6),
    )
end

function plot_square_hull!(plt; ns::Integer = 1_000, color=:bluesreds)
    x1 = x2 = range(-3, 3; length = ns)
    
    # y1 = 1
    objective1(x1, x2) = x1 - x2 + 1
    feasible1(x1, x2) = -2 ≤ x1 ≤ -1 && -2 ≤ x2 ≤ -1
    coloring1(x1, x2) = ifelse(feasible1(x1, x2), objective1(x1, x2), NaN)

    # y2 = 1
    objective2(x1, x2) = x1 - x2 + 5
    feasible2(x1, x2) = 1 ≤ x1 ≤ 2 && 1 ≤ x2 ≤ 2
    coloring2(x1, x2) = ifelse(feasible2(x1, x2), objective2(x1, x2), NaN)

    envelope(x1, x2) = (-1 ≤ x1 - x2 ≤ 1) && (-2 ≤ x1 ≤ 2) && (-2 ≤ x2 ≤ 2) 
    shading(x1, x2)  = ifelse(envelope(x1, x2), 1.0, NaN)

    heatmap!(
        plt, x1, x2, shading;
        color = :red,
        alpha = 0.2,
        colorbar_entry = false,
    )

    heatmap!(
        plt, x1, x2, coloring1;
        color = color,
    )

    heatmap!(
        plt, x1, x2, coloring2;
        color = color,
    )

    return plt
end

function plot_square_hull(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer = 1_000, plot_title="", feasible::Function=<(0), color=:bluesreds)
    plt = plot_square_hull_base()

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z  = reverse!([objective_value(model; result=i) for i = 1:nr])
    r  = reverse!([reads(model; result=i) for i = 1:nr])
    x  = collect.(zip(x1, x2))

    plot_square_hull!(plt; ns, color)

    plot_solutions!(plt, x, z, r, x⃰; feasible)

    plot!(plt; plot_title)

    return plt
end

function plot_square_indicator_base()
    plt = plot_base()

    return plot!(
        plt;
        title          = raw"$ \min~z = x_{1} - x_{2} + y_{1} + 5y_{2} $",
        colorbar_title = raw"$z$",
        xlims          = (-3, 3),
        ylims          = (-3, 3),
        clims          = ( 0, 6),
    )
end

function plot_square_indicator!(plt; ns::Integer = 1_000, color=:bluesreds)
    x1 = x2 = range(-3, 3; length = ns)
    
    # y1 = 1
    objective1(x1, x2) = x1 - x2 + 1
    feasible1(x1, x2) = -2 ≤ x1 ≤ -1 && -2 ≤ x2 ≤ -1
    coloring1(x1, x2) = ifelse(feasible1(x1, x2), objective1(x1, x2), NaN)

    # y2 = 1
    objective2(x1, x2) = x1 - x2 + 5
    feasible2(x1, x2) = 1 ≤ x1 ≤ 2 && 1 ≤ x2 ≤ 2
    coloring2(x1, x2) = ifelse(feasible2(x1, x2), objective2(x1, x2), NaN)

    envelope(x1, x2) = (-1 ≤ x1 - x2 ≤ 1) && (-2 ≤ x1 ≤ 2) && (-2 ≤ x2 ≤ 2) 
    shading(x1, x2)  = ifelse(envelope(x1, x2), 1.0, NaN)

    heatmap!(
        plt, x1, x2, coloring1;
        color = color,
    )

    heatmap!(
        plt, x1, x2, coloring2;
        color = color,
    )

    return plt
end

function plot_square_indicator(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer = 1_000, plot_title="", feasible::Function=<(0), color=:bluesreds)
    plt = plot_square_indicator_base()

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z  = reverse!([objective_value(model; result=i) for i = 1:nr])
    r  = reverse!([reads(model; result=i) for i = 1:nr])
    x  = collect.(zip(x1, x2))

    plot_square_indicator!(plt; ns, color)

    plot_solutions!(plt, x, z, r, x⃰; feasible)

    plot!(plt; plot_title)

    return plt
end

function solve_indint_squares(optimizer = DWave.Neal.Optimizer)
    return solve_indint_squares(identity, optimizer)
end

function solve_indint_squares(config!, optimizer = DWave.Neal.Optimizer)
    model = Model(() -> ToQUBO.Optimizer(optimizer))

    @variable(model, y[1:2], Bin)
    @variable(model, x[1:2])

    @objective(model, Min, x[1] - x[2] + y[1] + 5y[2])
    @constraint(model, exactly1, sum(y) == 1)

    let virtual_model = unsafe_backend(model)
        virtual_model.compiler_settings[:setup_callback] = (m::ToQUBO.Optimizer) -> begin
            n = 8  # number of bits
            e = ToQUBO.Encoding.Unary()
            S = [(-2.0, -1.0), (1.0, 2.0)]  # intervals
            W = Vector{ToQUBO.VI}(undef, 2) # Disjunction Variables

            for j = 1:2 # y indices
                W[j] = only(ToQUBO.Encoding.encode!(m, y[j].index, ToQUBO.Encoding.Mirror()).y)
            end

            for i = 1:2 # variables
                Z = ToQUBO.VI[]
                Ξ = ToQUBO.PBO.PBF{ToQUBO.VI,Float64}()
                Χ = nothing

                xi = x[i].index
                
                for j = 1:2 # disjuncts
                    # Manual encoding
                    z, ξ, χ = ToQUBO.Encoding.encode(e, S[j], n) do (nv::Union{Integer,Nothing} = nothing)
                        if isnothing(nv)
                            return MOI.add_variable(m.target_model)
                        else
                            return MOI.add_variables(m.target_model, nv)
                        end
                    end
                    
                    append!(Z, z)

                    Ξ += ToQUBO.PBO.PBF{ToQUBO.VI,Float64}(W[j]) * ξ

                    Χ = isnothing(χ) ? Χ : (isnothing(Χ) ? χ : (Χ += χ))
                end

                v = ToQUBO.Virtual.Variable{Float64}(
                    DisjunctEncoding(), # new encoding method
                    xi,
                    Z,
                    Ξ,
                    Χ,
                )

                ToQUBO.Encoding.encode!(m, v)
            end

            MOI.set(m, ToQUBO.Attributes.Quadratize(), true)
        end
    end

    config!(model)

    optimize!(model)

    return model
end
