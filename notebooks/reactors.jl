using Revise
using JuMP
using Plots
using DisjunctiveProgramming

includet("paper.jl")

# Set parameters
cx = [1.5, 1]
cy = [6.1, 5.9]
α = [0.75, 0.8]
d = 3 # not specified in the problem, I think it is 3

function solve_gdp_reactors(config!::Function, optimizer=HiGHS.Optimizer; method=Indicator())
    model = GDPModel(optimizer)

    @variable(model, 0 <= x[1:2] <= 5)
    @variable(model, Y[1:2], Logical)

    y = binary_variable.(Y)

    @objective(model, Min, cx'x + cy'y)

    @constraint(model, α'x ≥ d)

    @constraint(model, x[1] ≤ 0, Disjunct(Y[1]))
    @constraint(model, x[2] ≤ 0, Disjunct(Y[2]))

    disjunction(model, Y, exactly1=true)

    config!(model)

    optimize!(model; gdp_method=method)

    return model
end

solve_gdp_reactors(optimizer=HiGHS.Optimizer; method=Indicator()) = solve_gdp_reactors(identity, optimizer; method)

function plot_reactor()
    return plot(;
        size=(700, 600),
        title=raw"$ \min~z = \mathbf{c_{x}}'\mathbf{x} + \mathbf{c_{y}}'\mathbf{y} $",
        xlabel=raw"$x_1$",
        ylabel=raw"$x_2$",
        colorbar_title=raw"$z$",
        xlims=(-1, 6),
        ylims=(-1, 6),
        # clims          = ( 0, 6),
        legend = :outertop,
        aspect_ratio=:equal,
    )
end

function plot_reactor_feasible(; ns::Integer=1_000, color=:bluesreds)
    plt = plot_reactor()

    plot!(plt; plot_title="Feasible Region")

    return plot_reactor_feasible!(plt; ns, color)
end

function plot_reactor_feasible!(plt; ns::Integer=1_000, color=:bluesreds)
    x1 = filter(x -> α[1] * x ≥ d, range(0, 5; length=ns))
    x2 = filter(x -> α[2] * x ≥ d, range(0, 5; length=ns))
    xs = range(-1, 6; length=ns)

    plot!(
        plt,
        x1, zeros(length(x1));
        zcolor = (x1, x2) -> cx[1] * x1 + cy[2],
        label = "", # raw"$y_{2} = 1$",
        color,
        linewidth = 2,
        colorbar_entry = false,
    )

    plot!(
        plt,
        zeros(length(x2)), x2;
        zcolor = (x1, x2) -> cx[2] * x2 + cy[1],
        label = nothing, # raw"$y_{1} = 1$",
        color,
        linewidth = 2,
        colorbar_entry = false,
    )

    plot!(
        plt,
        xs, (x) -> (d - α[1] * x) / α[2];
        label     = raw"$\alpha'\mathbf{x} \geq d$",
        color     = :gray,
        linestyle = :dash,
    )

    return plt
end

function plot_reactor_optimal(x⃰; ns::Integer = 1_000, color=:bluesreds)
    plt = plot_reactor_feasible(; ns, color)

    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color  = :white,
        marker = :star8,
        markersize = 8,
        label  = "Optimal Solution",
    )

    return plt
end


function plot_reactor_bigm(
    model::JuMP.Model,
    x⃰::Vector{T},
    M::Number;
    nr::Integer=result_count(model),
    ns::Integer=1_000,
    color=:bluesreds,
) where {T}
    x = reverse!([value.(model[:x]; result=i) for i = 1:nr])
    z = reverse!([objective_value(model; result=i) for i = 1:nr])
    r = reverse!([reads(model; result=i) for i = 1:nr])

    return plot_reactor_bigm(x, z, r, x⃰, M; nr, ns, color)
end

function plot_reactor_bigm(x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; nr::Integer=length(x), ns::Integer=1_000, color=:bluesreds) where {T}
    plt = plot_reactor_feasible(; ns, color)

    return plot_reactor_bigm!(plt, x, z, r, x⃰, M; nr)
end

function plot_reactor_bigm!(plt, x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}, M::Number; nr::Integer=length(x)) where {T}
    plot!(plt; plot_title="Big-\$M\$ Feasible Region, \$M = $(M)\$, samples = $(nr)")

    plot_reactor_bigm_relaxation!(plt, M)

    return plot_solutions!(plt, x, z, r, x⃰; nr)
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
        color=:red,
        alpha=0.2,
        label=raw"Bigm relax",
        colorbar_entry=false,
    )

    return plt
end

function plot_reactor_hull(nr::Integer)
    return plot(;
        size=(700, 600),
        plot_title="Hull Feasible Region, samples = $(nr)",
        title=raw"$ \min~z = c_{x}'\mathbf{x} + c_{y}'\mathbf{y} $",
        xlabel=raw"$x_1$",
        ylabel=raw"$x_2$",
        xlims=(-1, 6),
        ylims=(-1, 6),
        # clims=( 0, 6),
        aspect_ratio=:equal,
        legend = :outertop,
        legend_columns=2,
        colorbar_title=raw"$z$",
    )
end

function plot_reactor_hull!(plt, x⃰, x; ns::Integer=1_000, color=:bluesreds)
    x1 = x2 = range(0, 5; length=ns)

    envelope(x1, x2) = (0 ≤ x2 + x1 ≤ 5) && (0 ≤ x1 ≤ 5) && (0 ≤ x2 ≤ 5) 
    shading(x1, x2)  = ifelse(envelope(x1, x2), 1.0, NaN)

    heatmap!(
        plt, x1, x2, shading;
        color = :red,
        alpha = 0.2,
        colorbar_entry = false,
        z_order = :back,
    )

    return plot_reactor_feasible!(plt; ns, color)
end

function plot_reactor_hull(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer=1_000, color=:bluesreds)
    plt = plot_reactor_hull(nr)

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z = reverse!([objective_value(model; result=i) for i = 1:nr])
    r = reverse!([reads(model; result=i) for i = 1:nr])
    x = [x1[end], x2[end]]

    scatter!(
        plt,
        x1,
        x2;
        color,
        zcolor=z,
        marker=:circle,
        markersize=4r,
        label="Samples",
        z_order=:front,
    )

    plot_reactor_hull!(plt, x⃰, x; ns, color)

    scatter!(
        plt,
        [x1[1]],
        [x[2]];
        color=:white,
        marker=:diamond,
        markersize=8,
        label="Best Sample",
    )

    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color=:white,
        marker=:star8,
        markersize=8,
        label="Optimal Solution",
    )

    return plt
end

function plot_reactor_indicator(nr::Integer)
    return plot(;
        size=(700, 600),
        plot_title="Indicator Feasible Region, samples = $(nr)",
        title=raw"$ \min~z = c_{x}'\mathbf{x} + c_{y}'\mathbf{y} $",
        xlabel=raw"$x_1$",
        ylabel=raw"$x_2$",
        xlims=(-1, 6),
        ylims=(-1, 6),
        # clims=(0, 6),
        aspect_ratio=:equal,
        legend = :outertop,
        legend_columns=2,
        colorbar_title=raw"$z$",
    )
end

function plot_reactor_indicator!(plt, x⃰, x; ns::Integer=1_000)
    # x1 = x2 = range(-3, 3; length=ns)

    plot_reactor_feasible!(plt; ns)

    scatter!(
        plt,
        [x[1]],
        [x[2]];
        color=:white,
        marker=:diamond,
        markersize=8,
        label="Best Sample",
    )

    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color=:white,
        marker=:star8,
        markersize=8,
        label="Optimal Solution",
    )

    plt
end

function plot_reactor_indicator(model::JuMP.Model, x⃰; nr::Integer=result_count(model), ns::Integer=1_000, color=:bluesreds)
    plt = plot_reactor_indicator(nr)

    x1 = reverse!([value(model[:x][1]; result=i) for i = 1:nr])
    x2 = reverse!([value(model[:x][2]; result=i) for i = 1:nr])
    z = reverse!([objective_value(model; result=i) for i = 1:nr])
    r = reverse!([reads(model; result=i) for i = 1:nr])
    x = [x1[end], x2[end]]

    scatter!(
        plt,
        x1,
        x2;
        color,
        zcolor=z,
        marker=:circle,
        markersize=4r,
        label="Samples",
        z_order=:front,
    )

    plot_reactor_indicator!(plt, x⃰, x; ns)

    plt
end
