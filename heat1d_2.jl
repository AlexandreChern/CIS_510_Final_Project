using SparseArrays
using Plots
using BenchmarkTools
include("naive_rk4_cuda.jl")


# u_t = α u_zz on 0 ≤ z ≤ depth_final
# u(x,0) = f(z)
# u(0,t) = g(t)
# u(depth_final, t) = f(depth_final)

# The initial condition will also serve as the at depth boundary condition
# geothermal in SI units
function init_cond(z)
    σ = -(1.8 * 10^(-6) * z.^2) /  (2 * (2))
    # Radiogenic Heat @ upper lithosphere * depth^2 / 2 * thermal conductivity of granidiorite
    η = (3.6 * 10^(-2) .* z) / 2
    # Heat flux @ crust/ mantle boundary * depth / thermal conductivity of granidiorite
    return σ + η .+ 288.0
    # λ + η + Average global surface temperature
end

function surf_bc(t, Δt)
    T_d = 15
    ω_d = Δt/(24)
    T_a = 30
    ω_a = Δt/(24*365)
    return 288 + (T_d * sin.(2*pi*ω_d*t)) # - T_a * sin.(ω_a*t))
end

#=
# Sample Function without source term
# u_t  = k u_xx + F(x,t) on 0 ≦ x ≦ 1
# u(x, 0) = f(x)
# u(0, t) = u(1, t) = 0
# k = 2
function f(x)
    return 6*sin.(pi*x)
end

function exact(x,t)
    return 6*sin.(pi*x)exp(-2*t*(pi)^2)
end
=#

function exact(t,Δt,z,α)
    T_d = 15
    ω_d = Δt/(24)
    T_a = 30
    ω_a = Δt/(24*365)
    return init_cond(z)+(exp.((-1).*z.*sqrt(2*pi*ω_d/(2α))).*T_d.*sin.(2*pi*ω_d.*t.-z.*sqrt(2*pi*ω_d/(2α))))#+exp.(-z.*sqrt(ω_a/(2α))).*sin.(ω_a.*t.-z.*sqrt(ω_a/(2α))))
end

function time_dependent_heat(k, Δz, Δt, tf ,t1, α, β, initial, exact, bound_cond, odesolve,num_th_block=0, num_block=0)
    N  = Integer(ceil((10-0)/Δz)) # N+1 total nodes, N-1 interior nodes
    z = 0:Δz:10
    t = 0:Δt:tf
    M = Integer(ceil((tf-0)/Δt)) # M+1 total temporal nodes

    # A is N+1 by N+1 because it contains boundary points

    A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N,1:N-1,ones(N-1),N+1,N+1) +
        sparse(2:N,3:N+1,ones(N-1),N+1,N+1))
    A[1,1]=(α/Δz^2)*1
    A[N+1,N+1]=(α/Δz^2)*1
    A[2,1]=(α/Δz^2)*1
    A[N,N+1]=(α/Δz^2)*1
    #println(A)
    # A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N+1,1:N,ones(N),N+1,N+1) + sparse(1:N,2:N+1,ones(N),N+1,N+1))

    u = Array{Float64}(zeros(N+1)) # Interior Nodes
    u .= exact(0,Δt,z[1:N+1],α)
    #println(u)

    #(t, U_inter, E_inter) = odesolve(Δt, t1, tf, u, A, my_exact)
    t2 = time()
    for i in 1:5
    (t, U, E) = odesolve(z, Δt, t1, tf, u, A, α, β, exact, bound_cond)
    end
    println("Time to solve on CPU:", (time() - t2)/5)
    (t, U, E) = odesolve(z, Δt, t1, tf, u, A, α, β, exact, bound_cond)
#=
    init = Array{Float64}(zeros(M+1,2))
    init[:,1] .= α
    init[:,2] .= β

    U .= U_inter#vcat(init[:,1]', U_inter, init[:,2]')
    E .= E_inter#vcat(init[:,1]', E_inter, init[:,2]')
=#
    return (z, t, U, E)

end



Δz_list = [2, 1, 0.5, 0.25, 0.125]
Δz_listt = [2,1,0.5,0.25]

function convergence_test()
    for Δz in Δz_listt
        k = 2.7
        Cp = 790
        ρ = 2700
        # Δz = 1
        @show Δz
        λ = 0.1
        Δt = round(λ*Δz^2, digits = 12)
        @show Δt
        #@assert 1 >= 2 * (k*Δt)/(Δx^2)
        tf = 480
        t1 = 0
        α = k/(ρ*Cp) #Thermal Diffusivity
        #α =
        #β =
        β = init_cond(10) #boundary condition
        N  = Integer(ceil((10-0)/Δz))

        num_th_blk = 32
        num_block = cld(N, num_th_blk)

        #(x, t, U, E) = time_dependent_heat(k, Δx, Δt, tf ,t1, α, β, f, exact, naive_rk4)
        (z, t, U, E) = time_dependent_heat(k, Δz, Δt, tf ,t1, α, β, init_cond, exact, surf_bc, naive_rk4, num_th_blk, num_block)

        #=
        xfine = 0:Δx/20:1

        stride_time = 100

        for i = 1:1:10

            p = plot(x,U[:,i],label = "approx", shape = :circle, color = :blue)
            ylims!((0,1))
            display(p)

            Efine = exact(xfine, t[i])
            pexact = plot!(xfine,Efine, label = "exact", color = :red)
            display(pexact)

            sleep(1)
        end

        =#

        @show err = sqrt(Δz) * norm(U[:,end]- E[:,end])
        @show (U[10:end,end] - E[10:end,end])
        @show rel_err = norm(U[:,end] - E[:,end])/norm(E[:,end])
        println(err)
        for i = size(U,2)
            p = plot(U[1:end,i],z[1:end],#=label = string("Numerical",string(Δt)),=# yflip = true, shape = :circle, color = :blue)
            #ylims!((0.0,1.0))
            xlabel!("Temperature (K)")
            ylabel!("Depth (m)")
            pexact = plot!(E[1:end,i],z[1:end], #= label = string("Exact",string(Δt)),=# yflip = true, color = :red)
            savefig(string("CPUplot",string(Δz),".png"))
        end
    end
end


function solve_GPU(k,Δz,Δt,t1,tf,α,β,exact,init_cond, bound_cond, num_th_block,num_block)
    N  = Integer(ceil((10-0)/Δz)) # N+1 total nodes, N-1 interior nodes
    z = 0:Δz:10
    t = 0:Δt:tf
    M = Integer(ceil((tf-0)/Δt)) # M+1 total temporal nodes

    # A is N+1 by N+1 because it contains boundary points

    A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N,1:N-1,ones(N-1),N+1,N+1) +
        sparse(2:N,3:N+1,ones(N-1),N+1,N+1))
    A[1,1]=(α/Δz^2)*1
    A[N+1,N+1]=(α/Δz^2)*1
    A[2,1]=(α/Δz^2)*1
    A[N,N+1]=(α/Δz^2)*1
    #println(A)
    # A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N+1,1:N,ones(N),N+1,N+1) + sparse(1:N,2:N+1,ones(N),N+1,N+1))

    # u = Array{Float64}(spzeros(N+1)) # All Nodes
    # u = randn(N+1)
    u = Array{Float64}(undef,N+1)
    u .= exact(0,Δt,z[1:N+1],α)

    #println(u)

    #(t, U_inter, E_inter) = odesolve(Δt, t1, tf, u, A, my_exact)
    # num_th_block = 32
    # num_block = cld(N, num_th_blk)
    t3 = time()
    for i in 1:5
    (all_t, cu_U) = cu_naive_rk4(z,Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_block, num_block)
    end
    println("Time to solve this problem on GPU with cu_naive_rk4 function call: ", (time() - t3)/5)
    (all_t, cu_U) = cu_naive_rk4(z,Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_block, num_block)
    return (all_t, cu_U)
end


function solve_GPU_new(k,Δz,Δt,t1,tf,α,β,exact,init_cond, bound_cond, num_th_blk,num_block)
    N  = Integer(ceil((10-0)/Δz)) # N+1 total nodes, N-1 interior nodes
    z = 0:Δz:10
    t = 0:Δt:tf
    M = Integer(ceil((tf-t1)/Δt)) # M+1 total temporal nodes

    # A is N+1 by N+1 because it contains boundary points

    t_ass = time()
    A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N,1:N-1,ones(N-1),N+1,N+1) +
        sparse(2:N,3:N+1,ones(N-1),N+1,N+1))
    A[1,1]=(α/Δz^2)*1
    A[N+1,N+1]=(α/Δz^2)*1
    A[2,1]=(α/Δz^2)*1
    A[N,N+1]=(α/Δz^2)*1
    println("Time to assemble A matrices ", time() - t_ass)
    #println(A)
    # A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N+1,1:N,ones(N),N+1,N+1) + sparse(1:N,2:N+1,ones(N),N+1,N+1))

    # u = Array{Float64}(spzeros(N+1)) # All Nodes
    # u = randn(N+1)
    u = Array{Float64}(undef,N+1)
    u .= exact(0,Δt,z[1:N+1],α)

    #println(u)

    #(t, U_inter, E_inter) = odesolve(Δt, t1, tf, u, A, my_exact)
    # num_th_block = 32
    # num_block = cld(N, num_th_blk)
    # (all_t, cu_U) = cu_naive_rk4(z,Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_block, num_block)

    d_U = CuArray{Float64}(undef,N+1,M+1)
    du = CuArray{Float64}(u)
    d_U[:,1] .= du

    t_s = time()
    dA = CuArray{Float64}(A)
    println("Time to convert A into CUDA array: ", time() - t_s)

    dy = CuArray{Float64}(undef,N+1)

    dy1 = similar(dy)
    dy2 = similar(dy)
    dy3 = similar(dy)

    u1_t_half = similar(dy)
    u2_t_half = similar(dy)
    u3 = similar(dy)

    t_start = time()
    for i in 1:5
    for n = 2:M+1
        # t = t + Δt
        @cuda threads = num_th_blk blocks = num_block knl_gemvs!(dA,du,dy)
        u1_t_half .= Δt/2 .* dy .+ du
        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u1_t_half,dy,dy1)
        u2_t_half .= Δt/2 .* dy1 .+ du
        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u2_t_half,dy,dy2)
        u3 .= Δt .* dy2 .+ du
        @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u3,dy,dy3)
        d_U[:,n] .= du .+ Δt/6 .* (dy .+ 2*dy1 .+ 2*dy2 .+ dy3)
    end
    end
    println("Time on the GPU kernels: ", (time() - t_start)/5)
    d_U[end,:] .= β
    d_U[1,:] = surf_bc.(t,Δt)
    return (t, d_U)
end

function solve_GPU_cusparse(k,Δz,Δt,t1,tf,α,β,exact,init_cond, bound_cond, num_th_block,num_block)
    N  = Integer(ceil((10-0)/Δz)) # N+1 total nodes, N-1 interior nodes
    z = 0:Δz:10
    t = 0:Δt:tf
    M = Integer(ceil((tf-t1)/Δt)) # M+1 total temporal nodes

    # A is N+1 by N+1 because it contains boundary points

    t_ass = time()
    A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N,1:N-1,ones(N-1),N+1,N+1) +
        sparse(2:N,3:N+1,ones(N-1),N+1,N+1))
    A[1,1]=(α/Δz^2)*1
    A[N+1,N+1]=(α/Δz^2)*1
    A[2,1]=(α/Δz^2)*1
    A[N,N+1]=(α/Δz^2)*1
    println("Time to assemble A matrices ", time() - t_ass)
    #println(A)
    # A = (α/Δz^2)*(-2 * sparse(1:N+1,1:N+1,ones(N+1),N+1,N+1) + sparse(2:N+1,1:N,ones(N),N+1,N+1) + sparse(1:N,2:N+1,ones(N),N+1,N+1))

    # u = Array{Float64}(spzeros(N+1)) # All Nodes
    # u = randn(N+1)
    u = Array{Float64}(undef,N+1)
    u .= exact(0,Δt,z[1:N+1],α)

    #println(u)

    #(t, U_inter, E_inter) = odesolve(Δt, t1, tf, u, A, my_exact)
    # num_th_block = 32
    # num_block = cld(N, num_th_blk)
    # (all_t, cu_U) = cu_naive_rk4(z,Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_block, num_block)

    d_U = CuArray{Float64}(undef,N+1,M+1)
    du = CuArray{Float64}(u)
    d_U[:,1] .= du

    t_s = time()
    dA_sparse = CuArrays.CUSPARSE.CuSparseMatrixCSC(A)
    println("Time to convert A into CUDA array: ", time() - t_s)

    dy = CuArray{Float64}(undef,N+1)

    dy1 = similar(dy)
    dy2 = similar(dy)
    dy3 = similar(dy)

    u1_t_half = similar(dy)
    u2_t_half = similar(dy)
    u3 = similar(dy)

    t_start = time()
    for i in 1:5
    for n = 2:M+1
        # t = t + Δt
        # @cuda threads = num_th_blk blocks = num_block knl_gemvs!(dA,du,dy)
        dy .= dA_sparse * du
        u1_t_half .= Δt/2 .* dy .+ du
        # @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u1_t_half,dy,dy1)
        dy1 .= dA_sparse * u1_t_half .+ dy
        u2_t_half .= Δt/2 .* dy1 .+ du
        # @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u2_t_half,dy,dy2)
        dy2 .= dA_sparse * u2_t_half .+ dy
        u3 .= Δt .* dy2 .+ du
        # @cuda threads = num_th_blk blocks = num_block knl_gemv!(dA,u3,dy,dy3)
        dy3 .= dA_sparse * u3 .+ dy
        d_U[:,n] .= du .+ Δt/6 .* (dy .+ 2*dy1 .+ 2*dy2 .+ dy3)
    end
    end
    println("Time on the GPU with Sparse Array Formulations: ", (time() - t_start)/5)
    d_U[end,:] .= β
    d_U[1,:] = surf_bc.(t,Δt)

    return (t, d_U)
end

let
    k = 2.7
    Cp = 790
    ρ = 2700
    Δz_list = [1,0.5,0.25,0.125,0.0625,0.03125, 0.015625]
    Δz_listt = [1,0.5]
    Δz_listtt = [0.25,0.125,0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]
    for Δz in Δz_listt
        @show Δz
        λ = 0.1
        # Δt = round(λ*Δz^2, digits = 12)
        Δt = round(λ*Δz, digits=12)
        @show Δt
        #@assert 1 >= 2 * (k*Δt)/(Δx^2)
        tf = 0.1
        # tf = 1
        t1 = 0
        α = k/(ρ*Cp) #Thermal Diffusivity
        #α =
        #β =
        β = init_cond(10) #boundary condition
        N  = Integer(ceil((10-0)/Δz))

        num_th_blk = 128
        num_block = cld(N, num_th_blk)
        @show (num_th_blk, num_block)

        # (all_t, cu_U) = solve_GPU(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)
        # @show Array(cu_U[:,end]) - cu_E[:,end]
        # diff = Array(cu_U[:,end]) - cu_E[:,end];|
        # @show diff[2]
        # @show diff[div(end,2)]

        # @show diff[div(end,10):9*div(end,10)]
        # @show norm(diff[div(end,10):9*div(end,10)])
        # (z, t, U, E) = time_dependent_heat(k, Δz, Δt, tf ,t1, α, β, init_cond, exact, surf_bc, naive_rk4, num_th_blk, num_block)
        # @show U[:,end] - E[:,end]
        println("This is GPU function that calles cu_naive_rk4: ")
        @time  solve_GPU(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)
        println()
        println("This is the GPU function that incorporate code from cu_naive_rk4: ")
        @time solve_GPU_new(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)
        println()
        println("This is direct solving on CPU: ")
        @time time_dependent_heat(k, Δz, Δt, tf ,t1, α, β, init_cond, exact, surf_bc, naive_rk4, num_th_blk, num_block)
        println()
        println("This is similar to CPU, but with sparse cuda array formulation: ")
        @time solve_GPU_cusparse(k,Δz,Δt,t1,tf,α,β,exact,init_cond, surf_bc, num_th_blk,num_block)
        println()
        println()
    end
end
