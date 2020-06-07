using LinearAlgebra
#using CUDAnative
#using CUDAdrv
#using CuArrays

function naive_rk4(Δt, t1, tf, u, A, exact)
    all_t = t1:Δt:tf
    t = t1
    N = length(u)
    M = Integer(ceil((tf - t1)/Δt))  # total number of points is N + 1 (Fence post problem)

    Exact = Matrix{Float64}(zeros(N,M+1))
    U = Matrix{Float64}(zeros(N,M+1))
    Exact[:,1] = u[:]
    U[:,1] = u[:]

    k = Matrix{Float64}(zeros(N,4))

        for n = 2:M+1
            t = t + Δt
            k[:,1] = A * u[:]
            u1_t_half = Δt/2 * k[:,1] + u[:]
            k[:,2] = A * u1_t_half
            u2_t_half = Δt/2 * k[:,2] + u[:]
            k[:,3] = A * u2_t_half
            u3 = Δt * k[:,3] + u[:]
            k[:,4] = A * u3
            u[:] = u[:] + Δt/6 * (k[:,1] + 2*k[:,2] + 2*k[:,3] + k[:,4])
            U[:,n] = u[:]
            Exact[:,n] = exact(t)
        end

            return (all_t, U, Exact)


end

function cu_naive_rk4(z, Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_blk, num_block)

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
    d_zero = CuArray(zeros(N))


    k = Matrix{Float64}(zeros(N,4))

        for n = 2:M+1

            t = t + Δt
            du = CuArray(u)
            dy = CuArray(hy)
            dy1 = CuArray(hy1)
            dy2 = CuArray(hy2)
            dy3 = CuArray(hy3)

            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du,d_zero,dy)
            k[:,1] = dy

            u1_t_half = Δt/2 * k[:,1] + u[:]
            du_half = CuArray(u1_t_half)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du_half,dy,dy1)
            k[:,2] = dy1

            u2_t_half = Δt/2 * k[:,2] + u[:]
            du2_half = CuArray(u2_t_half)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du2_half,dy,dy2)
            k[:,3] = dy2

            u3 = Δt * k[:,3] + u[:]
            du3 = CuArray(u3)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du3,dy,dy3)
            k[:,4] = dy3

            u[:] = u[:] + Δt/6 * (k[:,1] + 2*k[:,2] + 2*k[:,3] + k[:,4])
            U[:,n] = u[:]

            Exact[:,n] = exact(t,Δt,z,α)
        end

            return (all_t, U, Exact)


end
