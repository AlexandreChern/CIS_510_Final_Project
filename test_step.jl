using SparseArrays
using Plots
using BenchmarkTools
using CuArrays.CUSPARSE
include("naive_rk4_cuda.jl")

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

function exact(t,Δt,z,α)
    T_d = 15
    ω_d = Δt/(24)
    T_a = 30
    ω_a = Δt/(24*365)
    return init_cond(z)+(exp.((-1).*z.*sqrt(2*pi*ω_d/(2α))).*T_d.*sin.(2*pi*ω_d.*t.-z.*sqrt(2*pi*ω_d/(2α))))#+exp.(-z.*sqrt(ω_a/(2α))).*sin.(ω_a.*t.-z.*sqrt(ω_a/(2α))))
end

function solve_GPU(k,Δz,Δt,t1,tf,α,β,exact,init_cond, bound_cond, num_th_block,num_block)
    N  = Integer(ceil((10-0)/Δz)) # N+1 total nodes, N-1 interior nodes
    z = 0:Δz:10
    t = 0:Δt:tf
    M = Integer(ceil((tf-t1)/Δt)) # M+1 total temporal nodes

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
    t_start = time()
    for i in 1:10
    (all_t, cu_U) = cu_naive_rk4(z,Δt, t1, tf, u, A, α, β, exact, bound_cond, num_th_block, num_block)
    end
    println("Time to call cu_naive_rk4: ", (time() - t_start)/4)
    return (all_t, cu_U)
end

function solve_GPU_new(k,Δz,Δt,t1,tf,α,β,exact,init_cond, bound_cond, num_th_block,num_block)
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
    for i in 1:10
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
    println("Time on the GPU kernels: ", (time() - t_start)/10)
    d_U[end,:] .= β
    d_U[1,:] = surf_bc.(t,Δt)

    return (t, cu_U)
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
    for i in 1:10
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
    println("Time on the GPU with Sparse Array Formulations: ", (time() - t_start)/10)
    d_U[end,:] .= β
    d_U[1,:] = surf_bc.(t,Δt)

    return (t, d_U)
end

k = 2.7
Cp = 790
ρ = 2700

Δz = 0.0025

λ = 0.1
Δt = round(λ*Δz^2, digits = 12)
@show Δt
#@assert 1 >= 2 * (k*Δt)/(Δx^2)
tf = 0.01
# tf = 1
t1 = 0
α = k/(ρ*Cp) #Thermal Diffusivity
#α =
#β =
β = init_cond(10) #boundary condition
N  = Integer(ceil((10-0)/Δz))



# (all_t, cu_U) = solve_GPU(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)


############## Benchmark solve_GPU ##########################################
# @benchmark solve_GPU(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)




############## Benchmark Kernels ###########################################
N  = Integer(ceil((10-0)/Δz)) # N+1 total nodes, N-1 interior nodes|
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

d_A = CuArray(A)
U = Array{Float64}(undef,N+1,M+1)
d_U = CuArray{Float64}(undef,N+1,M+1)

# u = Array{Float64}(spzeros(N+1)) # All Nodes
# u = randn(N+1)
u = Array{Float64}(undef,N+1)
u .= exact(0,Δt,z[1:N+1],α)
du = CuArray(u)

#println(u)

bound_cond = surf_bc

#(t, U_inter, E_inter) = odesolve(Δt, t1, tf, u, A, my_exact)
# num_th_block = 32
# num_block = cld(N, num_th_blk)
# @benchmark (all_t, cu_U) = cu_naive_rk4(z,Δt, t1, tf, u, A, U, α, β, exact, bound_cond, num_th_blk, num_block)
#
#
# @benchmark naive_rk4(z, Δt, t1, tf, u, A, α, β, exact, bound_cond)

return (all_t, cu_U)

all_t = t1:Δt:tf
t = t1
N = length(u)
M = Integer(ceil((tf - t1)/Δt))

# @benchmark d_U = CuArray{Float64}(zeros(N,M+1))
# @benchmark du = CuArray{Float64}(u)
#
# @benchmark du = CuArray(u)
#
# @benchmark d_zero = CuArray{Float64}(zeros(N))
# d_zero = CuArray{Float64}(zeros(N))
# @benchmark dy1 = similar(d_zero)


# This just test assignment speed
# n_list = [10,10^2,10^3,10^4]
#
# for n in n_list
#     println()
#     println(n)
#     A = randn(n,n)
#     current_time = time()
#     for i in 1:10
#         cu_matrix = CuArray{Float64}(A)
#     end
#     time_last = time() - current_time
#     avg_time_1 = time_last/10
#     @show avg_time_1
#     println()
#     current_time_2 = time()
#     for i in 1:10
#         cu_undef = CuArray{Float64}(undef,n,n)
#     end
#     time_last_2 = time() - current_time_2
#     avg_time_2 = time_last_2/10
#     @show avg_time_2
# end


# cu_i = CuArray{Float64,1}(randn(3))
# cu_j = CuArray{Float64,1}(randn(3))
# cu_v = CuArray{Float64,1}(randn(3))
#
# i = [1,2,2,3]
# j = [1,2,3,3]
# v = [1,2,4,3]
#
# D_i = CuArray{Float64,1}(i)
# D_j = CuArray{Float64,1}(j)
# D_v = CuArray{Float64,1}(v)
#
# D_sparse = CUSPARSE.sparse(D_i,D_j,D_v)
#
# v = CuArray(randn(3))
# w = CuArray(randn(3))
#
# D_sparse * v
#
# D_sparse = CuArray(D_sparse)



# @time time_dependent_heat(k, Δz, Δt, tf ,t1, α, β, init_cond, exact, surf_bc, naive_rk4, num_th_blk, num_block)


num_th_blk = 32
num_block = cld(N, num_th_blk)
@show (num_th_blk, num_block)

# solve_GPU_new(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)

# println("Time to solve with GPU function call")
# @time solve_GPU(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)
println()
# println("Time for calling GPU kernel")
# @time solve_GPU_new(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)
println()
println("Time for CPU solve")
@time time_dependent_heat(k, Δz, Δt, tf ,t1, α, β, init_cond, exact, surf_bc, naive_rk4, num_th_blk, num_block)

@time solve_GPU_cusparse(k,Δz,Δt,t1,tf,α,β,exact, init_cond, surf_bc, num_th_blk, num_block)



######## This is a test on memory conversion ##########
using SparseArrays
using CuArrays
using BenchmarkTools

N = 100000
A_N = spzeros(N,N)

for i in 1:N
    A_N[i,i] = 1
end

for i in 1:N-1
    A_N[i,i+1] = -1
end


Base.summarysize(A_N)



cu_A_sparse = sparse(CuArray(A_N))

sizeof(cu_A_sparse)

cu_A_sparse_new = CuArrays.CUSPARSE.CuSparseMatrixCSC(A_N);

Base.summarysize(cu_A_sparse_new)

sizeof(cu_A_sparse_new)

b = randn(N)
cu_array = CuArray(b)

cu_A_sparse_new * cu_array

@benchmark cu_A_sparse_new * cu_array

@benchmark A_N * b

A_N * b - Array(cu_A_sparse_new * cu_array)

@benchmark A_N \ b
@benchmark cu_A_sparse_new \ cu_array
