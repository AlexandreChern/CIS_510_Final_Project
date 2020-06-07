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
