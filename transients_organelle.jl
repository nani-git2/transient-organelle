#=
Transient solutions to the organelle biogenesis problem, based on the method given in:
https://journals.aps.org/pre/pdf/10.1103/PhysRevE.91.062119

The organelle regulation has four possibilities:

                            Rate        Prefactor
1. De novo synthesis:       kd             -
2. Degradation      :       gamma          n
3. Fission          :       kfis           n
4. Fusion           :       kfus           n(n-1)

Depending on the rates and initial condition, any kind of special case can be considered.
=#

# Rate parameters
const kd = 8.0          # rate for de novo synthesis
const gamma = 10.0       # degradation
const kfis = 8.0       # Fission
const kfus = 0.0       # fussion

# Implementation parameters
const tau = 70         # cutoff for truncating the system of equations. 
const t = 0.0:0.01:10.0  # time array

# For various initial conditions. (Optional, can comment out)
const l = 10.0           # mean for Poisson distribution for initial values
const N = 4           # parameters for Binomial distribution
const p = 0.5


using StatsBase, LinearAlgebra, DifferentialEquations, Distributions, LaTeXStrings, SpecialFunctions 


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
Takes in the rates and tau, and returns the transition matrix eigenvalues, eig
"""
function matrix()
    a(m) = kd + (m-1)*kfis                      # a(m) goes from 1 to tau-1
    b(m) = (m+1)*gamma + m*(m+1)*kfus           # b(m) goes from 0 to tau-2

    dupper = Vector{Float64}(undef, tau-1)
    dlower = Vector{Float64}(undef, tau-1)
    diag = Vector{Float64}(undef, tau)
    for i in 1:(tau-1)
        dupper[i] = sqrt(b(i-1)*a(i))
        dlower[i] = sqrt(a(i)*b(i-1))
        diag[i] = -a(i) - b(i-2)
    end
    diag[1] = -a(1)
    diag[end] = -b(tau-2)           # no loss term for tau as ME itself is truncated
    A = Matrix(Tridiagonal(dlower, diag, dupper))   #transition matrix
    A = SymTridiagonal(A)
    eig = eigvals(A)                                # its eigenvalues
    return eig,A
end


"""
Storing the polynomials p and q as matrices. the i'th row corresponds to the polynomial p(i-1) and the j'th column in 
it corresponds to the value with j'th eigenvalue.

Similary, for q, the i'th row is the polynomial q(i+1), with the j'th column being for j'th eigenvalue.
"""
function poly(eig)
    p = Matrix{Float64}(undef, tau, tau)
    q = Matrix{Float64}(undef, tau, tau)
    a(m) = kd + (m-1)*kfis                      # a(m) goes from 1 to tau-1
    b(m) = (m+1)*gamma + m*(m+1)*kfus           # b(m) goes from 0 to tau-2
    p[1,:] = fill(1.0, tau)
    q[end,:] = fill(1.0, tau)

    for i in 1:tau
        p[2,i] = eig[i] + a(1)
        q[end-1,i] = eig[i] + b(tau-2)
    end

    for i in 3:tau
        for j in 1:tau
            p[i,j] = (eig[j]+a(i-1)+b(i-3))*p[i-1,j] - b(i-3)*a(i-2)*p[i-2,j]
        end
    end
    
    for i in tau-2:-1:1
        for j in 1:tau
            q[i,j] = (eig[j] + a(i+1) + b(i-1))*q[i+1,j] - b(i)*a(i+1)*q[i+2,j]
        end
    end
    return p,q
end


"""
Calculates Pm(t) for the given initial condition m0
"""
function single(m0)
    Pmt = Vector{Float64}(undef, length(t))         # P_m(t)
    a(m) = kd + (m-1)*kfis                      # a(m) goes from 1 to tau-1
    b(m) = (m+1)*gamma + m*(m+1)*kfus           # b(m) goes from 0 to tau-2
    eig = matrix()[1]
    p,q = poly(eig)
    pm = Matrix{Float64}(undef, tau, length(t))
    
    for m in 0:tau-1
        if m < m0
            function temp_low(r)
                s = 0.0
                for i in 1:tau
                    num = p[m+1,i]*q[m0+1,i]
                    den = 1.0
                    for j in 1:tau  
                        j != i ? den *= (eig[i] - eig[j]) : den *= 1.0
                    end
                    s += exp(eig[i]*r)*(num/den)
                end
                pre = 1.0
                for l in m:m0-1
                    pre *= b(l)
                end
                return s*pre
            end
            pm[m+1,:] = temp_low.(t)

        elseif m == m0
            function temp(r)
                s = 0.0
                for i in 1:tau
                    num = p[m+1,i]*q[m0+1,i]
                    den = 1.0
                    for j in 1:tau  
                        j != i ? den *= (eig[i] - eig[j]) : den *= 1.0
                    end
                    s += exp(eig[i]*r)*(num/den)
                end
                return s
            end
            pm[m+1,:] = temp.(t)

        elseif m > m0
            function temp_high(r)
                s = 0.0
                for i in 1:tau
                    num = p[m0+1,i]*q[m+1,i]
                    den = 1.0
                    for j in 1:tau  
                        j != i ? den *= (eig[i] - eig[j]) : den *= 1.0
                    end
                    s += exp(eig[i]*r)*(num/den)
                end
                pre = 1.0
                for l in m0+1:m
                    pre *= a(l)
                end
                return s*pre
            end
            pm[m+1,:] = temp_high.(t)
        end
    end
    return pm
end
        

"""
Main function, for a distribution of initial values.
"""
function main()
    Qm = initial("binomial")
    tot = zeros((tau, length(t)))
    for e in 1:length(Qm)
        if Qm[e] != 0
            tot += Qm[e]*single(e-1)
        end
    end
    return tot
end

"""
Evaluates the mean copy number and Fano factor as a function of time
"""
function moments(pm)
    mean_m = similar(t)
    x2_m = similar(t)
    for i in 1:length(t)
        s1 = 0.0
        s2 = 0.0
        for j in 0:tau-1
            s1 += j*pm[j+1,i]
            s2 += j*j*pm[j+1,i]
        end
        mean_m[i] = s1
        x2_m[i] = s2
    end
    var_m = x2_m .- (mean_m) .^ 2
    fano = var_m ./ mean_m
    return mean_m, fano
end


#To run, call output = moments(main()) in the Julia REPL, 
# or run the file from the shell after adding a line to store the data

