using LinearAlgebra
using CUDAnative
using CUDAdrv
using CuArrays

function knl_gemv!(A,x,b,y)
    (M,N) = size(A)
    len = length(A)
    @assert M == N;
    bid = blockIdx().x
    tid = threadIdx().x
    dim = blockDim().x

    y .= 0

    i = dim * (bid - 1) + tid

    if i <= M
        for k=1:N
            y[i] += A[i,k]*x[k]
        end
        y[i] += b[i]
    end

    return nothing
end

function my_forward_Euler(x, Δt, t1, tf, A, y1, N, M, exact, surf_bc, α, β)

    l = length(y1)
    U = zeros(Float64,l,M+1)
    E = zeros(Float64,l,M+1)
    C = zeros(Float64,l,M+1)
    y = copy(y1)      # initial guess
    U[:,1] = y[:]
    t=t1
    println(size(A),size(y))

    for n = 2:M+1
        t=t+Δt
        y[:] += Δt*(A*y[:])
        U[:,n] = y[:]
        E[:,n] = exact(t,Δt,x[:],α)
        y[1] = surf_bc(t, Δt)
        y[end] = β
    end

    return (U, E, t)

end

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
            u[end] = β
            uk = u
            k[:,1] = A * u[:]
            uk[1] = bound_cond(t+Δt/2, Δt)
            uk[end] = β
            u1_t_half = Δt/2 * k[:,1]
            k[:,2] = k[:,1] + A * u1_t_half
            uk[1] = bound_cond(t+Δt, Δt)
            uk[end] = β
            u2_t_half = Δt/2 * k[:,2]
            k[:,3] = k[:,1] + A * u2_t_half
            uk[1] = bound_cond(t+2*Δt, Δt)
            uk[end]=β
            u3 = Δt * k[:,3]
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

function cu_naive_rk4(z, Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_blk, num_block)

    all_t = t1:Δt:tf
    t = t1
    N = length(u)
    M = Integer(ceil((tf - t1)/Δt))

    Exact = Matrix{Float64}(zeros(N,M+1))
    # U = Matrix{Float64}(zeros(N,M+1))
    # U = zeros(N,M+1)
    Exact[:,1] = u[:]
    # U[:,1] = u[:]
    d_U = CuArray{Float64}(zeros(N,M+1))
    du = CuArray{Float64}(u)
    d_U[:,1] .= du[:]


    # hy = zeros(N)
    # hy1 = zeros(N)
    # hy2 = zeros(N)
    # hy3 = zeros(N)
    dA = CuArray(A)
    d_zero = CuArray(zeros(N))


    # k = Matrix{Float64}(zeros(N,4))
    # d_k = CuArray{Float64}(zeros(N,4))

    # dy = CuArray(spzeros(N))
    # dy1 = CuArray(spzeros(N))
    # dy2 = CuArray(spzeros(N))
    # dy3 = CuArray(spzeros(N))

    dy = similar(d_zero)
    # dy1 = CuArray(zeros(N))
    # dy2 = CuArray(zeros(N))
    # dy3 = CuArray(zeros(N))
    dy1 = similar(dy)
    dy2 = similar(dy)
    dy3 = similar(dy)

    u1_t_half = similar(dy)
    u2_t_half = similar(dy)
    u3 = similar(dy)

    for n = 2:M+1
        t = t + Δt
        # du = CuArray(u)
        # dy = CuArray(hy)
        # dy1 = CuArray(hy1)
        # dy2 = CuArray(hy2)
        # dy3 = CuArray(hy3)

        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du,d_zero,dy)
        # d_k[:,1] = dy

        # u1_t_half = Δt/2 * d_k[:,1] + du[:]
        u1_t_half .= Δt/2 * dy .+ du[:]
        # du_half = CuArray(u1_t_half)
        # @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du_half,dy,dy1)
        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u1_t_half,dy,dy1)
        # d_k[:,2] = dy1

        # u2_t_half = Δt/2 * d_k[:,2] + du[:]
        u2_t_half .= Δt/2 * dy1 .+ du[:]
        # du2_half = CuArray(u2_t_half)
        # @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du2_half,dy,dy2)
        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u2_t_half,dy,dy2)
        # d_k[:,3] = dy2

        # u3 = Δt * d_k[:,3] + du[:]
        u3 .= Δt * dy2 .+ du[:]
        # du3 = CuArray(u3)
        # @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du3,dy,dy3)
        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u3,dy,dy3)
        # d_k[:,4] = dy3

        # du[:] = du[:] + Δt/6 * (d_k[:,1] + 2*d_k[:,2] + 2*d_k[:,3] + d_k[:,4])
        # u = Array(du)
        # u = Array(du[:] + Δt/6 * (d_k[:,1] + 2*d_k[:,2] + 2*d_k[:,3] + d_k[:,4]))
        # u = Array(du[:] + Δt/6 * (dy + 2*dy1 + 2*dy2 + dy3))
        # u = du[:] + Δt/6 * (dy + 2*dy1 + 2*dy2 + dy3)
        # @show typeof(u)
        # u[1] = bound_cond(t, Δt)
        # u[end]=β
        d_U[:,n] .= du[:] .+ Δt/6 * (dy .+ 2*dy1 .+ 2*dy2 .+ dy3)
        # d_U[1,n] = bound_cond(t, Δt)
        # d_U[end,n] = β
        # d_U[:,n] = u[:]

        Exact[:,n] = exact(t,Δt,z,α)
    end
    d_U[end,:] .= β
    d_U[1,:] = surf_bc.(all_t,Δt)
    return (all_t, d_U, Exact)
end




######################################## OLD CODE ####################################












######################################### OLD CODE ##############################












function cu_naive_rk4_old(z, Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_blk, num_block)

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
            k[:,1] .= dy

            u1_t_half = Δt/2 * k[:,1] + u[:]
            du_half = CuArray(u1_t_half)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du_half,dy,dy1)
            k[:,2] .= dy1

            u2_t_half = Δt/2 * k[:,2] + u[:]
            du2_half = CuArray(u2_t_half)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du2_half,dy,dy2)
            k[:,3] .= dy2

            u3 = Δt * k[:,3] + u[:]
            du3 = CuArray(u3)
            @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,du3,dy,dy3)
            k[:,4] = dy3

            u[:] = u[:] + Δt/6 * (k[:,1] + 2*k[:,2] + 2*k[:,3] + k[:,4])
            # @show typeof(u)
            u[1] = bound_cond(t, Δt)
            u[end]=β
            U[:,n] = u[:]

            Exact[:,n] = exact(t,Δt,z,α)
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
