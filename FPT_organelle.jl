#=
Organelle number regulation has four possibilities:

                            Rate        Prefactor
1. De novo synthesis:       kd             -
2. Degradation      :       gamma          n
3. Fission          :       kfis           n
4. Fusion           :       kfus           n(n-1)

Depending on the rates and initial condition, any kind of special case can be considered.
=#

#=
Gillespie algorithm for the organelle biogenesis problem.

Four possible processes:

                            Rate        Prefactor
1. De novo synthesis:       kd             -
2. Degradation      :       gamma          n
3. Fission          :       kfis           n
4. Fusion           :       kfus           n(n-1)

=#

using StatsBase, Random, Distributions, LinearAlgebra


# Rate parameters
const kd = 10.0          # rate for de novo synthesis
const gamma = 1.0       # degradation
const kfis = 0.0       # Fission
const kfus = 0.0       # fussion

# Target copy numbers
const Xvals = Vector(1:1:30)

# For various initial conditions. (Optional, can comment out)
const l = 10.0           # mean for Poisson distribution for initial values
const N = 4           # parameters for Binomial distribution
const p = 0.5


"""
Defining the initial condition. Some common examples are given. 
However, one can use an emperically obtained initial distribution
for Qm (just make sure that it is normalized).

Also, while using the delta function for initial condition (all cells have same copy number),
note the the fission-fusion process has to start from 1, not 0.

Mixed refers to taking subpopulations with different distributions.
"""
function initial(key) :: Vector{Float64}
    if key == "poisson"
        d = Poisson(l)
        x = Vector(0:1:tau-1)
        Qm = pdf.(d, x)
        return Qm
    elseif key == "delta"
        Qm = zeros(tau)
        Qm[2] = 1.0
        return Qm
    elseif key == "uniform"
        Qm = ones(tau)
        Qm = Qm/sum(Qm)
        return Qm
    elseif key == "binomial"
        Qm = zeros(tau)
        d = Binomial(N,p)
        x = Vector(0:1:tau-1)
        Qm = pdf.(d, x)
        Qm = Qm/sum(Qm)
    elseif key == "mixed"
        d1 = Poisson(l)
        d2 = Binomial(N,p)
        x = Vector(0:1:tau-1)
        Qm1 = pdf.(d1, x)
        Qm2 = pdf.(d2, x)
        ssfrac = 0.75
        Qm = ssfrac*Qm1 + (1-ssfrac)*Qm2 
        Qm = Qm/sum(Qm)
        return Qm
    end
end




"""
Takes in the rates and X, and returns the transition matrix
"""
function matrix_all(X,  kd::Float64, gamma::Float64, kfis::Float64, kfus::Float64)
    a(m) = kd + (m-1)*kfis                      # a(m) goes from 1 to xval-1
    b(m) = (m+1)*gamma + m*(m+1)*kfus           # b(m) goes from 0 to xval-2

    dupper = Vector{Float64}(undef, X-1)
    dlower = Vector{Float64}(undef, X-1)
    diag = Vector{Float64}(undef, X)
    for i in 1:(X-1)
        dupper[i] = b(i-1)
        dlower[i] = a(i)
        diag[i] = -a(i) - b(i-2)
    end
    diag[1] = -a(1)
    diag[end] = -a(X)-b(X-2)
    A = Matrix(Tridiagonal(dlower, diag, dupper))   #transition matrix
    return A
end


"""
Compute the moments numerically, for arbitary initial conditions
"""
function numerical(X::Int64)
    U = zeros(X)
    U[end] = kd + (X-1)*kfis
    P0 = initial("delta", X)
    A = matrix_all(X)
    mfpt = (U')*((inv(A))^2)*P0
    secmom = -2*(U')*((inv(A))^3)*P0
    cv = (sqrt(secmom - mfpt^2))/mfpt
    return mfpt, cv
end
