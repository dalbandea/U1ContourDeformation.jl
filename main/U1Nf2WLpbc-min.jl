# U1 Nf2
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using BDIO
using LinearAlgebra
using FormalSeries
using U1ContourDeformation

ENV["JULIA_DEBUG"] = "all"

# Generation of configurations

beta = 5.0
lsize = 2
mass = 0.6

model = U1Nf2(Float64, beta = beta, iL = (lsize, lsize), am0 = mass, BC = PeriodicBC)

model = U1Quenched(Float64, beta = beta, iL = (lsize, lsize), BC = PeriodicBC)

samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 5)))

LFTU1.randomize!(model)

@time sample!(model, samplerws)

Ss = Vector{Float64}(undef, 10000)

for i in 1:10000
    @time sample!(model, samplerws)
    # Ss[i] = gauge_action(model)
end

fname = "run-U1-pbc-b$beta-L$lsize-nc10000.bdio"
fb = BDIO_open(fname, "w","U1 Quenched")
BDIO_close!(fb)

fname = "run-pbc-b$beta-L$lsize-nc10000.bdio"
fb = BDIO_open(fname, "w","U1 Quenched")
BDIO_close!(fb)

# Thermalize
for i in 1:10000
    @time sample!(model, samplerws)
end

# Run
for i in 1:10000
    @time sample!(model, samplerws)
    fb = BDIO_open(fname, "a","U1 Quenched")
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 1, true)
    BDIO_write!(fb,model)
    BDIO_write_hash!(fb)
    BDIO_close!(fb)
end

fname = "run-pbc-b$beta-L$lsize-nc10000.bdio"

fname = "run-U1-pbc-b$beta-L$lsize-nc10000.bdio"

ens = [deepcopy(model) for i in 1:10000]
read_ensemble!(ens, fname)


# Check minimization of the action {{{

beta = 5.0
lsize = 2
mass = 0.2

model = U1Nf2(
              Float64, 
              # Series{ComplexF64, 2},
              beta = beta, 
              iL = (lsize, lsize), 
              am0 = mass, 
              BC = PeriodicBC
             )

LFTU1.randomize!(model)

function compute_action(delta)
    U = copy(model.U) .* exp.(im*delta) .* exp.(-im*delta)
    Umodel = U1Nf2(
                  Float64, 
                  custom_init = U,
                  beta = beta, 
                  iL = (lsize, lsize), 
                  am0 = mass, 
                  BC = PeriodicBC
                 )
    Umodel.U .*= exp.(im*delta)
    return gauge_action(Umodel)
end

delta = zeros(Float64, size(model.U))

ders = ForwardDiff.gradient(compute_action, delta)
delta .-= 0.01 .* ders
Utest = deepcopy(model)
Utest.U .*= exp.(im*delta)
gauge_action(Utest)

model.U .* exp.(im*delta)

# }}}

# Check minimization of Dirac determinant {{{


function compute_detDwsr(delta)
    U = copy(model.U) .* exp.(im*delta) .* exp.(-im*delta)
    Umodel = U1Nf2(
                  Float64, 
                  custom_init = U,
                  beta = beta, 
                  iL = (lsize, lsize), 
                  am0 = mass, 
                  BC = PeriodicBC
                 )
    Umodel.U .*= exp.(im*delta)
    x1 = similar(U)
    x2 = similar(U)
    xtmp = similar(U)
    D = similar(real(U), prod(size(U)), prod(size(U)))
    x1 .= 0.0
    x2 .= 0.0
    xtmp .= 0.0
    for i in eachindex(x1)
        x1[i] = 1.0
        gamm5Dw_sqr_msq!(x2, xtmp, x1, Umodel)
        xtmp .= 0.0
        for j in eachindex(xtmp)
            xtmp[j] = 1.0
            D[i,j] = real(dot(x2, xtmp))
            xtmp[j] = 0.0
        end
        x1[i] = 0.0
    end
    return det(D/maximum(D))
end


model = U1Nf2(
              Float64, 
              # Series{ComplexF64, 2},
              beta = beta, 
              iL = (lsize, lsize), 
              am0 = mass, 
              BC = PeriodicBC
             )

LFTU1.randomize!(model)

delta = zeros(Float64, size(model.U))

delta[1,1,1] = 0.2

ders = ForwardDiff.gradient(compute_detDwsr, delta)
delta .-= 100 .* ders
ders

Utest = deepcopy(model)
Utest.U .*= exp.(im*delta)
gauge_action(Utest)

compute_detDwsr(delta)

# }}}

# Test 1 without fermion determinant => Bias result {{{

using Statistics

Ss = gauge_action.(ens)

var(Ss)


function compute_var(ensemble, delta)
    S = Vector{Any}(undef, length(ensemble))
    for i in eachindex(ensemble)
        println(i)
        U = copy(ens[i].U) .* exp.(im*delta) .* exp.(-im*delta)
        Umodel = U1Nf2(
                       Float64, 
                       custom_init = U,
                       beta = beta, 
                       iL = (lsize, lsize), 
                       am0 = mass, 
                       BC = PeriodicBC
                      )
        S_undef = gauge_action(Umodel)
        Umodel.U .*= exp.(-delta)
        S_def = gauge_action(Umodel)
        obs = gauge_action(Umodel)
        S[i] = exp(-S_undef + S_def) * obs
    end
    println(mean(S))
    return var(real.(S))
end

using ForwardDiff

compute_vars_comp(x) = compute_var(ens, x)

delta = zeros(Float64, size(model.U))

ForwardDiff.gradient(x -> compute_var(ens, x), delta)

ders = ForwardDiff.gradient(compute_vars_comp, delta)
delta .-= 0.1 * ders
ders

compute_vars_comp(delta)

# }}}

# Test 2: minimize action variance of quenched theory {{{

using Statistics

using ADerrors

Ss = gauge_action.(ens)

id = "test"

uws = uwreal(Ss, id)
uwerr(uws)
uws

taui(uws, id)

var(Ss)

function compute_var(ensemble, delta)
    S = Vector{Any}(undef, length(ensemble))
    for i in eachindex(ensemble)
        # println(i)
        U = copy(ensemble[i].U) .* exp.(im*delta) .* exp.(-im*delta)
        Umodel = U1Quenched(
                       Float64, 
                       custom_init = U,
                       beta = beta, 
                       iL = (lsize, lsize), 
                       BC = PeriodicBC
                      )
        S_undef = action(Umodel)
        Umodel.U .*= exp.(-delta)
        S_def = action(Umodel)
        obs = action(Umodel)
        S[i] = exp(-S_undef + S_def) * obs
    end
    println(mean(S))
    return var(real.(S))
end

using ForwardDiff

compute_vars_comp(x) = compute_var(ens, x)

delta = zeros(Float64, size(model.U))

ForwardDiff.gradient(x -> compute_var(ens, x), delta)

ders = ForwardDiff.gradient(compute_vars_comp, delta)
delta[1,1,1] -= 0.01 * ders[1,1,1]
# delta .-= 0.1 * ders
ders

compute_vars_comp(delta)

# }}}

# Test 2: minimize action variance  with fermion determinant {{{

using Statistics

Ss = gauge_action.(ens)

var(Ss)

function compute_Dwsr!(D, x1, x2, xtmp, xmodel)
    D .= 0.0
    x1 .= 0.0
    x2 .= 0.0
    xtmp .= 0.0
    for i in eachindex(x1)
        x1[i] = 1.0
        gamm5Dw_sqr_msq!(x2, xtmp, x1, xmodel)
        xtmp .= 0.0
        for j in eachindex(xtmp)
            xtmp[j] = 1.0
            D[i,j] = real(dot(x2, xtmp))
            xtmp[j] = 0.0
        end
        x1[i] = 0.0
    end
end


function compute_var(ensemble, delta)
    S = Vector{Any}(undef, length(ensemble))
    x1 = similar(copy(ensemble[1].U) .* exp.(im*delta) .* exp.(-im*delta))
    x2 = similar(x1)
    xtmp = similar(x1)
    D = similar(x1, prod(size(x1)), prod(size(x1)))
    for i in eachindex(ensemble)
        # println(i)
        U = copy(ensemble[i].U) .* exp.(im*delta) .* exp.(-im*delta)
        Umodel = U1Nf2(
                       Float64, 
                       custom_init = U,
                       beta = beta, 
                       iL = (lsize, lsize), 
                       am0 = mass, 
                       BC = PeriodicBC
                      )
        S_undef = gauge_action(Umodel)
        compute_Dwsr!(D, x1, x2, xtmp, Umodel)
        # maxD_undef = maximum(D)
        detD_undef = det(D)
        Umodel.U .*= exp.(-delta)
        S_def = gauge_action(Umodel)
        obs = gauge_action(Umodel)
        compute_Dwsr!(D, x1, x2, xtmp, Umodel)
        # maxD_def = maximum(D)
        detD_def = det(D)
        S[i] = exp(-S_undef + S_def) #=*exp(prod(size(x1))*(log(maxD_undef/maxD_def)))=# * detD_undef / detD_def * obs
    end
    println(mean(S))
    return var(real.(S))
end

using ForwardDiff

compute_vars_comp(x) = compute_var(ens, x)

delta = zeros(Float64, size(model.U))

ForwardDiff.gradient(x -> compute_var(ens, x), delta)

ders = ForwardDiff.gradient(compute_vars_comp, delta)
delta[1,1,1] -= 0.01 * ders[1,1,1]
# delta .-= 0.1 * ders
ders

compute_vars_comp(delta)

# }}}
