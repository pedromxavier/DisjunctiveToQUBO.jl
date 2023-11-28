include("paper.jl")

using JuMP
using Plots
using DisjunctiveProgramming

# function solve_ind_squares(config!::Function, optimizer=HiGHS.Optimizer; method=Indicator())
#     model = Model(optimizer)

#     @variable(model, -2 <= x[1:2] <= 2)
#     @variable(model, y[1:2], Bin)

#     @objective(model, Min, x[1] - x[2] + y[1] + 5y[2])

#     @constraint(model, sum(y) == 1)

#     @constraint(model, sq1[i=1:2], y[1] => {-2 ≤ x[i] ≤ -1})
#     @constraint(model, sq2[i=1:2], y[2] => {1 ≤ x[i] ≤ 2})

#     config!(model)

#     optimize!(model)

#     return model
# end

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

function plot_square()
    return plot(;
        size           = (700, 600),
        title          = raw"$ \min~z = x_{1} - x_{2} + y_{1} + 5y_{2} $",
        xlabel         = raw"$x_1$",
        ylabel         = raw"$x_2$",
        colorbar_title = raw"$z$",
        xlims          = (-3, 3),
        ylims          = (-3, 3),
        clims          = ( 0, 6),
        aspect_ratio   = :equal,
    )
end

function plot_square_feasible(; ns::Integer=1_000, color=:balance)
    plt = plot_square()

    return plot_square_feasible!(plt; ns, color)
end

function plot_square_feasible!(plt; ns::Integer=1_000, color=:balance)
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
    color=:balance,
) where {T}
    x = reverse!([value.(model[:x]; result=i) for i = 1:nr])
    z = reverse!([objective_value(model; result=i) for i = 1:nr])
    r = reverse!([reads(model; result=i) for i = 1:nr])

    return plot_square_bigm(x, z, r, x⃰, M; nr, ns, color)
end

function plot_square_bigm(x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; nr::Integer = length(x), ns::Integer=1_000, color=:balance) where {T}
    plt = plot_square_feasible(; ns, color)

    return plot_square_bigm!(plt, x, z, r, x⃰, M; nr)
end

function plot_square_bigm!(plt, x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; nr::Integer=length(x)) where {T}
    plot!(plt; plot_title="Big-\$M\$ Feasible Region, \$M = $(M)\$, samples = $(nr)")

    plot_square_bigm_relaxation!(plt, M)

    return plot_solutions!(plt, x, z, r, x⃰; nr)
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

function plot_square_hull(nr::Integer)
    return plot(;
        size         = (700, 600),
        plot_title   = "Hull Feasible Region, samples = $(nr)",
        title        = raw"$ \min~z = x_{1} - x_{2} + y_{1} + 5y_{2} $",
        xlabel       = raw"$x_1$",
        ylabel       = raw"$x_2$",
        xlims        = (-3, 3),
        ylims        = (-3, 3),
        clims        = ( 0, 6),
        aspect_ratio = :equal,
        legend_columns = 2,
        colorbar_title = raw"$z$",
    )
end

function plot_square_hull!(plt, x⃰, x; ns::Integer = 1_000)
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
        xlims = extrema(x1),
        ylims = extrema(x2),
        colorbar_entry = false,
    )

    heatmap!(
        plt, x1, x2, coloring1;
        xlims = extrema(x1),
        ylims = extrema(x2),
        z_order = :back,
    )

    heatmap!(
        plt, x1, x2, coloring2;
        xlims = extrema(x1),
        ylims = extrema(x2),
        z_order = :back,
    )

    scatter!(
        plt,
        [x[1]],
        [x[2]];
        color  = :white,
        marker = :circle,
        markersize = 8,
        label  = "Best Sample",
    )

    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color  = :white,
        marker = :star8,
        markersize = 8,
        label  = "Optimal Solution",
    )

    plt
end

function plot_square_hull(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer = 1_000)
    plt = plot_square_hull(nr)

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z  = reverse!([objective_value(model; result=i) for i = 1:nr])
    r  = reverse!([reads(model; result=i) for i = 1:nr])
    x  = [x1[end], x2[end]]

    scatter!(
        plt,
        x1,
        x2;
        zcolor     = z,
        marker     = :circle,
        markersize = 4r,
        label      = "Samples",
        z_order    = :front,
    )

    plot_square_hull!(plt, x⃰, x; ns)

    plt
end

function plot_square_indicator(nr::Integer)
    return plot(;
        size         = (700, 600),
        plot_title   = "Indicator Feasible Region, samples = $(nr)",
        title        = raw"$ \min~z = x_{1} - x_{2} + y_{1} + 5y_{2} $",
        xlabel       = raw"$x_1$",
        ylabel       = raw"$x_2$",
        xlims        = (-3, 3),
        ylims        = (-3, 3),
        clims        = ( 0, 6),
        aspect_ratio = :equal,
        legend_columns = 2,
        colorbar_title = raw"$z$",
    )
end

function plot_square_indicator!(plt, x⃰, x; ns::Integer = 1_000)
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

    # heatmap!(
    #     plt, x1, x2, shading;
    #     color = :red,
    #     alpha = 0.2,
    #     xlims = extrema(x1),
    #     ylims = extrema(x2),
    #     colorbar_entry = false,
    # )

    heatmap!(
        plt, x1, x2, coloring1;
        xlims = extrema(x1),
        ylims = extrema(x2),
        z_order = :back,
    )

    heatmap!(
        plt, x1, x2, coloring2;
        xlims = extrema(x1),
        ylims = extrema(x2),
        z_order = :back,
    )

    scatter!(
        plt,
        [x[1]],
        [x[2]];
        color  = :white,
        marker = :circle,
        markersize = 8,
        label  = "Best Sample",
    )

    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color  = :white,
        marker = :star8,
        markersize = 8,
        label  = "Optimal Solution",
    )

    plt
end

function plot_square_indicator(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer = 1_000)
    plt = plot_square_indicator(nr)

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z  = reverse!([objective_value(model; result=i) for i = 1:nr])
    r  = reverse!([reads(model; result=i) for i = 1:nr])
    x  = [x1[end], x2[end]]

    scatter!(
        plt,
        x1,
        x2;
        zcolor     = z,
        marker     = :circle,
        markersize = 4r,
        label      = "Samples",
        z_order    = :front,
    )

    plot_square_indicator!(plt, x⃰, x; ns)

    plt
end
