########################################################################
# Weighted-SSA / (Q,R)-Cadzow (Gillard & Zhigljavsky, 2016)
# Panel-wise re-fit (Q,R) + FIXED number of global passes per panel.
# Plot & RMSE: EXCLUDE ONLY THE LAST MONTH (keep reconstruction unchanged).
########################################################################

using LinearAlgebra, Statistics
using JuMP, OSQP
using CSV, DataFrames
using Plots
using Dates
using Measures 

# --------------------------- Utilities --------------------------------

function hankelize(y::AbstractVector, L::Int)
    N = length(y) - 1
    @assert 0 ≤ L < length(y)
    K = N - L
    X = Matrix{Float64}(undef, L+1, K+1)
    @inbounds for i in 0:L, j in 0:K
        X[i+1, j+1] = y[i+j+1]
    end
    return X
end

function dehankel_weighted(X::AbstractMatrix, q::AbstractVector, r::AbstractVector)
    Lp1, Kp1 = size(X)
    @assert length(q) == Lp1
    @assert length(r) == Kp1
    L, K = Lp1-1, Kp1-1
    N = L + K
    y = zeros(Float64, N+1)
    w = zeros(Float64, N+1)
    @inbounds for i in 0:L, j in 0:K
        n = i + j
        wij = q[i+1]*r[j+1]
        w[n+1] += wij
        y[n+1] += wij*X[i+1, j+1]
    end
    @inbounds for n in 0:N
        y[n+1] = y[n+1] / w[n+1]
    end
    return y
end

function lowrank_QR(X::AbstractMatrix, q::AbstractVector, r::AbstractVector, rnk::Int)
    Lp1, Kp1 = size(X)
    @assert length(q) == Lp1
    @assert length(r) == Kp1
    rnk ≤ 0 && return zero(X)

    Qs = Diagonal(sqrt.(q))
    Rs = Diagonal(sqrt.(r))
    Y  = (Qs * X) * Rs

    sv = svd(Y; full=false)
    rkeep = min(rnk, length(sv.S))
    rkeep == 0 && return zero(X)

    U_r = sv.U[:, 1:rkeep]
    S_r = Diagonal(sv.S[1:rkeep])
    V_r = Matrix(sv.V[:, 1:rkeep])

    Qsinv = Diagonal(1.0 ./ sqrt.(q))
    Rsinv = Diagonal(1.0 ./ sqrt.(r))
    return (Qsinv * (U_r * S_r)) * (V_r') * Rsinv
end

function conv_design_R(W::AbstractVector, q::AbstractVector, L::Int)
    N = length(W)-1
    K = N - L
    A = zeros(Float64, N+1, K+1)
    @inbounds for n in 0:N
        kmin = max(0, n - L); kmax = min(K, n)
        for k in kmin:kmax
            A[n+1, k+1] = q[n - k + 1]
        end
    end
    return A
end

function conv_design_Q(W::AbstractVector, r::AbstractVector, L::Int)
    N = length(W)-1
    K = N - L
    A = zeros(Float64, N+1, L+1)
    @inbounds for n in 0:N
        lmin = max(0, n - K); lmax = min(L, n)
        for l in lmin:lmax
            A[n+1, l+1] = r[n - l + 1]
        end
    end
    return A
end

function fit_QR_from_W(W::AbstractVector; L::Int, α::Real=0.1, ϵ::Real=1e-6,
                       maxiters::Int=500, n_starts::Int=1000)
    N = length(W)-1
    @assert 0 ≤ L ≤ N
    K = N - L

    dist_best = Inf
    q_best = Vector{Float64}()
    r_best = Vector{Float64}()

    for _ in 1:n_starts
        q = α .+ (1 .- α) .* rand(L+1);  q[1] = 1.0
        r = α .+ (1 .- α) .* rand(K+1)

        last_q = copy(q)
        for _iter in 1:maxiters
            A_r = conv_design_R(W, q, L)
            model = Model(OSQP.Optimizer); set_silent(model)
            @variable(model, rvar[1:K+1] >= α)
            @objective(model, Min, sum((A_r * rvar .- W).^2))
            optimize!(model)
            r = value.(rvar)

            A_q = conv_design_Q(W, r, L)
            model2 = Model(OSQP.Optimizer); set_silent(model2)
            @variable(model2, qvar[1:L+1] >= α)
            @constraint(model2, qvar[1] == 1.0)
            @objective(model2, Min, sum((A_q * qvar .- W).^2))
            optimize!(model2)
            q = value.(qvar)

            if norm(q - last_q) ≤ ϵ
                break
            end
            last_q .= q
        end

        Wr = zeros(Float64, N+1)
        @inbounds for n in 0:N
            lmin = max(0, n - K); lmax = min(L, n)
            acc = 0.0
            for l in lmin:lmax
                acc += q[l+1]*r[n-l+1]
            end
            Wr[n+1] = acc
        end
        dist = sum((W .- Wr).^2)

        if dist < dist_best
            dist_best = dist
            q_best = copy(q); r_best = copy(r)
        end
    end

    return q_best, r_best, dist_best
end

# Run exactly `passes` global Cadzow updates (no convergence test)
# λ controls how strongly we "follow" observed data:
#   λ = 1.0 -> original behavior (hard clamping, y[i]=y0[i] on observed)
#   λ = 0.0 -> full global fit (no clamping, whole graph can move)
#   0 < λ < 1 -> soft pull towards observed data
function cadzow_QR_passes(y0::AbstractVector, miss_mask::BitVector,
                          q::AbstractVector, r::AbstractVector;
                          L::Int, rnk::Int, passes::Int,
                          λ::Real = 1.0)
    y = copy(y0)
    @inbounds for _ in 1:passes
        X  = hankelize(y, L)
        Xr = lowrank_QR(X, q, r, rnk)
        y  = dehankel_weighted(Xr, q, r)

        if λ ≥ 1
            # original hard clamping on observed entries
            for i in eachindex(y)
                if !miss_mask[i]
                    y[i] = y0[i]
                end
            end
        elseif λ ≤ 0
            # full global fit: do nothing, keep low-rank estimate everywhere
            nothing
        else
            # soft pull towards observed data
            for i in eachindex(y)
                if !miss_mask[i]
                    y[i] = λ * y0[i] + (1 - λ) * y[i]
                end
            end
        end
    end
    return y, passes
end

# ---------------------- Data loader (Fortified) ------------------------

function _parse_month_any(x)::Date
    if x isa Date
        return Date(year(x), month(x), 1)
    end
    s = String(x)
    for fmt in (dateformat"y-m-d", dateformat"y-m", dateformat"u-yy", dateformat"u-yyyy",
                dateformat"m/y", dateformat"m/y/u")
        try
            d = Date(s, fmt)
            return (year(d) < 200) ? Date(year(d)+1900, month(d), 1) : Date(year(d), month(d), 1)
        catch
        end
    end
    error("Unrecognized date format for Month/date value: $s")
end

function load_fortified(; csv_path::AbstractString)
    @assert isfile(csv_path) "CSV not found at: $csv_path"
    df0 = CSV.read(csv_path, DataFrame)
    lowernames = lowercase.(names(df0))

    if ("month" in lowernames) && ("fortified" in lowernames)
        month_col = names(df0)[findfirst(==("month"), lowernames)]
        fort_col  = names(df0)[findfirst(==("fortified"), lowernames)]
        dates = _parse_month_any.(df0[!, month_col])
        fortified = Float64.(df0[!, fort_col])
        df = DataFrame(date=dates, fortified_thousands_litres=fortified)
    elseif ("date" in lowernames) && ("fortified_thousands_litres" in lowernames)
        date_col = names(df0)[findfirst(==("date"), lowernames)]
        fort_col = names(df0)[findfirst(==("fortified_thousands_litres"), lowernames)]
        df = DataFrame(date=Date.(df0[!, date_col]), fortified_thousands_litres=Float64.(df0[!, fort_col]))
    else
        error("Unsupported CSV schema. Expected (Month, Fortified, …) or (date, fortified_thousands_litres).")
    end

    mask = (df.date .>= Date(1980,1,1)) .& (df.date .<= Date(1991,1,1))
    return df[mask, :]
end

# ---------------------- Fortified-wine experiment ----------------------

function run_fortified_pipeline(; L=36, rnk=11, α=0.1, ϵ=1e-6, W_missing=0.3,
                                n_starts_QR=1000, λ_cadzow::Real=0.0,
                                csv_path::AbstractString)

    df = load_fortified(csv_path=csv_path)
    y_true_full = Vector{Float64}(df.fortified_thousands_litres)
    dates = Vector{Date}(df.date)

    N0 = 121
    @assert length(y_true_full) ≥ N0 + 12
    y_work = copy(y_true_full)

    # --- masking indices for reconstruction (UNCHANGED) ---
    idx_mid = 60:71
    idx_end = (N0):(N0+12)      # 13-point end block as in your code

    # --- EXCLUDE last month from plots and RMSE only ---
    Nplot        = N0 + 12 - 1            # show/evaluate up to penultimate month
    idx_total    = 1:Nplot
    idx_end_eval = first(idx_end):(last(idx_end)-1)

    miss_mask = falses(length(y_work))
    miss_mask[idx_mid] .= true
    miss_mask[idx_end] .= true

    # Figure 4
    fig4a = plot(dates[1:N0], y_true_full[1:N0], lw=2, xlabel="Date", ylabel="Thousand litres",
                 title="Figure 4(a): Complete series (Jan 1980 – Jan 1990)", legend=false)

    y_with_gaps = copy(y_work); y_with_gaps[idx_mid] .= NaN; y_with_gaps[idx_end] .= NaN
    fig4b = plot(dates[1:Nplot], y_with_gaps[1:Nplot], lw=2, xlabel="Date", ylabel="Thousand litres",
                 title="Figure 4(b): Series with middle gap + end block", legend=false)

    # Entry weights
    W = ones(Float64, length(y_work)); W[miss_mask] .= W_missing

    # Mean fill start
    y_curr = copy(y_work); μ = mean(y_work[.!miss_mask]); y_curr[miss_mask] .= μ

    # Fixed passes per panel (tweak as desired)
    panel_passes = (1, 1, 1, 6)

    yhats = Vector{Vector{Float64}}(undef, 4)
    labels = ["Povprečje", "Ena iteracija", "Dve iteraciji", "Pet iteracij"]
    used_passes = Int[]

    # (a)
    q_a, r_a, _ = fit_QR_from_W(W; L=L, α=α, ϵ=ϵ, n_starts=n_starts_QR)
    y_a, p_a    = cadzow_QR_passes(y_curr, miss_mask, q_a, r_a;
                                   L=L, rnk=rnk, passes=panel_passes[1],
                                   λ=λ_cadzow)
    yhats[1] = y_a; push!(used_passes, p_a)

    # (b)
    q_b, r_b, _ = fit_QR_from_W(W; L=L, α=α, ϵ=ϵ, n_starts=n_starts_QR)
    y_b, p_b    = cadzow_QR_passes(y_a, miss_mask, q_b, r_b;
                                   L=L, rnk=rnk, passes=panel_passes[2],
                                   λ=λ_cadzow)
    yhats[2] = y_b; push!(used_passes, p_b)

    # (c)
    q_c, r_c, _ = fit_QR_from_W(W; L=L, α=α, ϵ=ϵ, n_starts=n_starts_QR)
    y_c, p_c    = cadzow_QR_passes(y_b, miss_mask, q_c, r_c;
                                   L=L, rnk=rnk, passes=panel_passes[3],
                                   λ=λ_cadzow)
    yhats[3] = y_c; push!(used_passes, p_c)

    # (d)
    q_d, r_d, _ = fit_QR_from_W(W; L=L, α=α, ϵ=ϵ, n_starts=n_starts_QR)
    y_d, p_d    = cadzow_QR_passes(y_c, miss_mask, q_d, r_d;
                                   L=L, rnk=rnk, passes=panel_passes[4],
                                   λ=λ_cadzow)
    yhats[4] = y_d; push!(used_passes, p_d)

    println("Global passes used per panel: ", used_passes)
    println("λ_cadzow = $λ_cadzow (0.0 = full global fit, 1.0 = hard clamp to observed data)")

    # RMSEs (exclude last month in End & Total)
    metrics = DataFrame(
        StartingValues = labels,
        Middle = [sqrt(mean((yhats[k][idx_mid]      .- y_true_full[idx_mid]).^2)) for k in 1:4],
        End    = [sqrt(mean((yhats[k][idx_end_eval] .- y_true_full[idx_end_eval]).^2)) for k in 1:4],
        Total  = [sqrt(mean((yhats[k][idx_total]    .- y_true_full[idx_total]).^2)) for k in 1:4],
    )

    # Figure 5 panels (plot up to Nplot)
    figs5 = Vector{Any}()
    for k in 1:length(labels)
        p = plot(dates[1:Nplot], y_true_full[1:Nplot], ls=:dash, lw=2, label="Opazovano",
                 xlabel="Datum", ylabel="Tisoč litrov",
                 title="$(labels[k])")
        plot!(p, dates[1:Nplot], yhats[k][1:Nplot], lw=2, label="Aproksimacija")
        push!(figs5, p)
    end

    return (q_panels=(q_a,q_b,q_c,q_d), r_panels=(r_a,r_b,r_c,r_d),
            W=W, yhats=yhats, metrics=metrics,
            fig4a=fig4a, fig4b=fig4b, figs5=figs5, passes=used_passes,
            Nplot=Nplot)
end

# ---------------------- Example run ------------------------------------

res = run_fortified_pipeline(; L=36, rnk=11, α=0.1, ϵ=1e-6,
                             W_missing=0.3, n_starts_QR=1,
                             λ_cadzow=0.8,      # 0.0 -> run on the whole graph
                             csv_path="AustralianWine.csv")

show(res.metrics, allrows=true, allcols=true)

# Display individual figures (optional, for interactive use)
display(res.fig4a); display(res.fig4b)
for p in res.figs5
    display(p)
end

# 2×2 grid of the four panel plots (optional Figure 5 grid)
pgrid = plot(res.figs5...; layout=(2,2), size=(1200,800), margin=5mm)
savefig(pgrid, "fig5_panels.pdf")

# ------------------- Last four figures in one vertical PDF -------------------

# res.figs5 already contains the last four panel plots
# (Povprečje, Ena iteracija, Dve iteraciji, Pet iteracij)

p_last4 = plot(
    res.figs5...;
    layout = (length(res.figs5), 1),        # 4 rows, 1 column
    size   = (900, 350 * length(res.figs5)),
    margin = 5mm
)

# Save only these four, one under another, into a single PDF
savefig(p_last4, "fortified_last4_vertical.pdf")
