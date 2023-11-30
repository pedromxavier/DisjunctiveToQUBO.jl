function plot_base()
    return plot(;
        size           = (1000, 900),
        xlabel         = raw"$x_1$",
        ylabel         = raw"$x_2$",
        aspect_ratio   = :equal,
        legend         = :outerbottom,
        legend_columns = -1,
    )
end

function plot_infeasible_solutions!(plt, x::Vector{Vector{T}}, s::Vector{T}; infcolor = :darkred) where {T}
    scatter!(
        plt,
        [x[i][1] for i = 1:length(x)],
        [x[i][2] for i = 1:length(x)];
        color             = :darkmagenta,
        markershape       = :diamond,
        markersize        = 20s,
        markerstrokewidth = 0.5,
        label             = "Infeasible",
    )

    return plt
end

function plot_feasible_solutions!(plt, x::Vector{Vector{T}}, z::Vector{T}, s::Vector{T}; color = :bluesreds) where {T}
    scatter!(
        plt,
        [x[i][1] for i = 1:length(x)],
        [x[i][2] for i = 1:length(x)];
        zcolor            = z,
        color             = color,
        marker            = :circle,
        markersize        = 20s,
        markerstrokewidth = 0.25,
        label             = "Feasible",
    )

    return plt
end

function plot_optimal_solution!(plt, x⃰)
    scatter!(
        plt,
        [x⃰[1]],
        [x⃰[2]];
        color             = :white,
        marker            = :star8,
        markersize        = 10,
        markerstrokewidth = 0.5,
        label             = "Optimal",
    )

    return plt
end

function plot_best_sample!(plt, x::Vector{Vector{T}}) where {T}
    scatter!(
        plt,
        [x[end][1]],
        [x[end][2]];
        color             = :darkcyan,
        marker            = :rect,
        markersize        = 10,
        markerstrokewidth = 0.5,
        label             = "Best Found",
    )

    return plt
end

function plot_solutions!(plt, x::Vector{Vector{T}}, z::Vector{T}, r::Vector{Int}, x⃰::Vector{T}; color=:bluesreds, infcolor=:darkred, feasible::Function = (z) -> (z < 0.0)) where {T}
    R = maximum(r)
    s = r ./ R

    ji = [i for i = 1:length(z) if !feasible(z[i])]
    jf = [i for i = 1:length(z) if  feasible(z[i])]
    
    if !isempty(ji)
        xi = [x[j] for j in ji]
        si = [s[j] for j in ji]
        plot_infeasible_solutions!(plt, xi, si; infcolor)
    end

    if !isempty(jf)
        xf = [x[j] for j in jf]
        zf = [z[j] for j in jf]
        sf = [s[j] for j in jf]

        plot_feasible_solutions!(plt, xf, zf, sf; color)
    end

    plot_best_sample!(plt, x)

    return plot_optimal_solution!(plt, x⃰)
end
