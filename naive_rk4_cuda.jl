using LinearAlgebra
#using CUDAnative
#using CUDAdrv
#using CuArrays

function naive_rk4(z, Δt, t1, tf, u, A, α, β, exact, bound_cond)
    all_t = t1:Δt:tf
    t = t1
    N = length(u)
    M = Integer(ceil((tf - t1)/Δt))  # total number of points is N + 1 (Fence post problem)

    Exact = Matrix{Float64}(zeros(N,M+1))
    U = Matrix{Float64}(zeros(N,M+1))
    Exact[:,1] = u[:]
    U[:,1] = u[:]
    #println(U[:,1])

    k = Matrix{Float64}(zeros(N,4))

        for n = 2:M+1
            t = t + Δt
            u[1] = bound_cond(t, Δt)
            u[end]=β
            uk = u
            k[:,1] = A * u[:]
            uk[1] = bound_cond(t+Δt/2, Δt)
            uk[end]=β
            u1_t_half = Δt/2 * uk[:]
            k[:,2] = k[:,1] + A * u1_t_half
            uk[1] = bound_cond(t+Δt, Δt)
            uk[end]=β
            u2_t_half = Δt/2 * uk[:]
            k[:,3] = k[:,1] + A * u2_t_half
            uk[1] = bound_cond(t+2*Δt, Δt)
            uk[end]=β
            u3 = Δt * uk[:]
            k[:,4] = k[:,1] + A * u3
            u[:] = u[:] + Δt/6 * (k[:,1] + 2*k[:,2] + 2*k[:,3] + k[:,4])
            u[1] = bound_cond(t, Δt)
            u[end]=β
            U[:,n] = u[:]
            #println(u[end])
            Exact[:,n] = exact(t, Δt, z, α)
            #println(sqrt(Δz) * norm(U[1:50,n]- Exact[1:50,n]))
        end

            return (all_t, U, Exact)


end
#=
function cu_naive_rk4(Δt, t1, tf, u, A, exact, num_th_blk, num_block)

    all_t = t1:Δt:tf
    t = t1
    N = length(u)
    M = Integer(ceil((tf - t1)/Δt))

    Exact = Matrix{Float64}(zeros(N,M+1))
    U = Matrix{Float64}(zeros(N,M+1))
    Exact[:,1] = u[:]
    U[:,1] = u[:]

    hy = zeros(N)
    hy1 = zeros(N)
    hy2 = zeros(N)
    hy3 = zeros(N)
    dA = CuArray(A)


    k = Matrix{Float64}(zeros(N,4))

        for n = 2:M+1

            t = t + Δt
            du = CuArray(u)
            dy = CuArray(hy)
            dy1 = CuArray(hy1)
            dy2 = CuArray(hy2)
            dy3 = CuArray(hy3)

            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dy,dA,du)
            k[:,1] = dy

            u1_t_half = Δt/2 * k[:,1] + u[:]
            du_half = CuArray(u1_t_half)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dy1,dA,du_half)
            k[:,2] = dy1

            u2_t_half = Δt/2 * k[:,2] + u[:]
            du2_half = CuArray(u2_t_half)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dy2,dA,du2_half)
            k[:,3] = dy2

            u3 = Δt * k[:,3] + u[:]
            du3 = CuArray(u3)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dy3,dA,du3)
            k[:,4] = dy3

            u[:] = u[:] + Δt/6 * (k[:,1] + 2*k[:,2] + 2*k[:,3] + k[:,4])
            U[:,n] = u[:]

            Exact[:,n] = exact(t)
        end

            return (all_t, U, Exact)


end

function knl_gemv!(y,A,u,b)

    (M, N) = size(A)

    @assert length(y) == M
    @assert length(u) == N
    @assert length(b) == M

    bidx = blockIdx().x  # get the thread's block ID
    tidx = threadIdx().x # get my thread ID
    dimx = blockDim().x  # how many threads in each block

    i = dimx * (bidx - 1) + tidx

    if i <= M
        for k = 1:N
            y[i] += A[i, k] * u[k]
            if k == N
                y[i] +=b[i]
        end
    end
    return nothing
end
=#
