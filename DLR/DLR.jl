using LinearAlgebra, Statistics, Random, Plots

# --- 1D Damped Rank Reduction (SSA-style) ---

# Construct a Hankel matrix from a 1D vector d.
function hankel_matrix(d::Vector{T}) where T
    L = length(d)
    M = fld(L, 2) + 1                  # ≈ L/2 + 1, near-square window
    N = L - M + 1
    H = Matrix{float(T)}(undef, M, N)  # use floating container (handles Int/Complex)
    @inbounds for i in 1:M, j in 1:N
        H[i, j] = d[i + j - 1]
    end
    return H
end

# Inverse Hankel operator: reconstruct a vector by averaging along anti-diagonals.
function inv_hankel(H::AbstractMatrix)
    M, N = size(H)
    L = M + N - 1
    T = float(eltype(H))
    d_est = zeros(T, L)              # ensure floating division
    counts = zeros(Int, L)
    @inbounds for i in 1:M, j in 1:N
        k = i + j - 1
        d_est[k] += H[i, j]
        counts[k] += 1
    end
    @inbounds for k in 1:L
        d_est[k] /= counts[k]
    end
    return d_est
end

# Damped TSVD: SVD, truncate to rank K, apply damping operator.
function damped_tsvd(M::AbstractMatrix, K::Int, damp::Real)
    F = svd(M; full=false)                # safe: returns U, S, Vt
    r = min(K, length(F.S))
    S1 = F.S[1:r]
    S2 = F.S[r+1:end]
    δ̂ = isempty(S2) ? zero(eltype(S1)) : maximum(S2)

    # Guard against tiny/zero singular values; build damping vector and clamp to [0,1]
    S1safe = max.(S1, eps(eltype(S1)))
    t = 1 .- (δ̂^damp) .* (S1safe .^ (-damp))   # entries of T on the diagonal
    t = clamp.(t, 0, 1)
    Td = Diagonal(t)

    U1  = F.U[:, 1:r]
    Σ1  = Diagonal(S1)
    V1t = F.Vt[1:r, :]                         # this is V1' already
    return U1 * (Σ1 * Td) * V1t
end

# 1D Damped Rank Reduction: Hankelization → damped TSVD → inverse Hankel.
function damped_rank_reduction_1d(d::Vector, K::Int, damp::Real)
    D = hankel_matrix(d)
    M_damped = damped_tsvd(D, K, damp)
    d_est = inv_hankel(M_damped)
    return d_est
end

# Function to compute SNR: 10*log10(||C||^2 / ||C - A||^2)
compute_snr(clean::Vector, estimated::Vector) =
    10 * log10(sum(abs2, clean) / sum(abs2, clean .- estimated))

# --- Test the algorithm on a 1D signal ---

# Define the time axis and a clean test signal.
t = 1:50
clean_signal = [sin(0.2 * i) for i in t]

# Add random Gaussian noise.
rng = MersenneTwister(1234)
noise = 0.2 .* randn(rng, length(clean_signal))
noisy_signal = clean_signal .+ noise

# Set parameters for damped rank reduction.
K = 2      # target rank (real sinusoid → rank 2)
damp = 2   # damping exponent (N)

# Compute runtime and apply the damped rank reduction algorithm.
elapsed_time = @elapsed denoised_signal = damped_rank_reduction_1d(noisy_signal, K, damp)

# Compute SNR of the denoised signal.
snr_value = compute_snr(clean_signal, denoised_signal)
println("SNR: ", snr_value, " dB")
println("Elapsed time: ", elapsed_time, " seconds")

# Plot the clean, noisy, and denoised signals.
plot(t, clean_signal, label="Čisti signal", lw=2)
plot!(t, noisy_signal, label="Šumni signal", lw=2, ls=:dash)
plot!(t, denoised_signal, label="Očiščen signal (Damped)", lw=2, ls=:dot)
xlabel!("Čas")
ylabel!("Amplituda")
title!("Damped Rank Reduction for 1D Signal")


#Primer za 2d signal


# using LinearAlgebra, Statistics, Random, Plots

# # ---------------------------
# # 1. Utility functions for 1D signals
# # ---------------------------

# # Construct a Hankel matrix from a 1D vector d.
# function hankel_matrix(d::Vector{T}) where T
#     L = length(d)
#     M = floor(Int, L/2) + 1
#     N = L - M + 1
#     H = zeros(T, M, N)
#     for i in 1:M
#         for j in 1:N
#             H[i, j] = d[i + j - 1]
#         end
#     end
#     return H
# end

# # Inverse Hankel operator: reconstruct a 1D vector by averaging along the anti-diagonals.
# function inv_hankel(H::AbstractMatrix{T}) where T
#     M, N = size(H)
#     L = M + N - 1
#     d_est = zeros(T, L)
#     counts = zeros(Int, L)
#     for i in 1:M
#         for j in 1:N
#             k = i + j - 1
#             d_est[k] += H[i, j]
#             counts[k] += 1
#         end
#     end
#     for k in 1:L
#         d_est[k] /= counts[k]
#     end
#     return d_est
# end

# # Damped TSVD: perform SVD, truncate to rank K, and then apply a damping operator.
# #
# # In this implementation, after computing the SVD:
# #    M = U * diagm(S) * V'
# # we let S1 = S[1:K] and S2 = S[K+1:end] (if any).
# # We then compute δ̂ = maximum(S2) (or zero if S2 is empty) and define a damping operator
# # T_damp = I - diagm( S1.^(-damp) * δ̂ ).
# # Finally, the damped low-rank approximation is:
# #
# #    M_damped = U[:,1:K] * diagm(S1) * T_damp * V[:,1:K]'.
# function damped_tsvd(M::AbstractMatrix{T}, K::Int, damp::Real) where T
#     U, S, V = svd(M)
#     K = min(K, length(S))
#     S1 = S[1:K]
#     S2 = S[K+1:end]
#     δ̂ = isempty(S2) ? zero(eltype(S)) : maximum(S2)
#     T_damp = Diagonal(ones(eltype(S1), K) .- (S1 .^ (-damp)) * δ̂)
#     M_damped = U[:, 1:K] * Diagonal(S1) * T_damp * V[:, 1:K]'
#     return M_damped
# end

# # Damped MSSA denoising for a 1D signal.
# function damped_mssa_denoise(d::Vector{T}, R::Int, damp::Real) where T
#     D = hankel_matrix(d)
#     M_damped = damped_tsvd(D, R, damp)
#     d_est = inv_hankel(M_damped)
#     return d_est
# end

# # ---------------------------
# # 2. Extend to 2D data: apply damped MSSA row-wise
# # ---------------------------
# function damped_mssa_denoise_2d(S::AbstractMatrix{T}, R::Int, damp::Real) where T
#     Ny, Nx = size(S)
#     S_est = similar(S)
#     for i in 1:Ny
#         row = S[i, :]
#         denoised_row = damped_mssa_denoise(row, R, damp)
#         # The inverse Hankel operator may return a longer vector;
#         # here, we take only the first Nx elements.
#         S_est[i, :] = denoised_row[1:Nx]
#     end
#     return S_est
# end

# # ---------------------------
# # 3. Generate 2D test signal and add noise
# # ---------------------------
# function generate_test_signal(Ny::Int, Nx::Int)
#     S = zeros(Float64, Ny, Nx)
#     for i in 1:Ny
#         for j in 1:Nx
#             S[i, j] = sin(0.2 * j + 0.1 * i) + 0.5*sin(0.15 * j - 0.05 * i)
#         end
#     end
#     return S
# end

# Ny = 50
# Nx = 50
# S_clean = generate_test_signal(Ny, Nx)

# rng = MersenneTwister(1234)
# noise_level = 0.2
# S_noisy = S_clean .+ noise_level .* randn(rng, Ny, Nx)

# # ---------------------------
# # 4. Apply damped MSSA denoising row-wise.
# # ---------------------------
# R = 2        # Target rank
# damp = 1.0   # Damping factor

# S_denoised = damped_mssa_denoise_2d(S_noisy, R, damp)

# # ---------------------------
# # 5. Plot the results together in one figure.
# # ---------------------------
# p = plot(
#     heatmap(S_clean, title="Čisti signal", colorbar=true),
#     heatmap(S_noisy, title="Šumni signal", colorbar=true),
#     heatmap(S_denoised, title="Očiščen signal (damped MSSA)", colorbar=true),
#     layout = (1, 3), size=(1200, 400)
# )
# display(p)
