using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using BDIO
using LinearAlgebra
using FormalSeries
using U1ContourDeformation
using Statistics

using ADerrors
id = "test"

ENV["JULIA_DEBUG"] = "all"

fname = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/run-b5.555-L2-nc10000.bdio"


beta = 5.555
lsize = 2
model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

ens = [deepcopy(model) for i in 1:10000]

read_ensemble!(ens, fname)

Ss = gauge_action.(ens)

mean(Ss)

function W1(model)
    U = model.U
    return real(cos(-im*U[1,1,1] * U[2,1,2] / U[1,2,1] / U[1,1,2]))
end


W1s = real.(W1.(ens))

uwW1 = uwreal(W1s, id)
uwerr(uwW1)
uwW1


# Deforming all links

using ReverseDiff
import ReverseDiff: gradient

delta1 = 0.0
delta2 = 0.0
delta3 = 0.0
delta4 = 0.0

deltas = (delta1, delta2, delta3, delta4)

links = [angle.([ei.U[1,1,1],ei.U[2,1,2],ei.U[1,2,1],ei.U[1,1,2]]) for ei in ens]

function W1link(lplaqs)
    return lplaqs[1] + lplaqs[2] - lplaqs[3] - lplaqs[4]
end

function W1link_action(lplaqs, beta)
    theta = W1link(lplaqs)
    return -beta*cos.(theta)
end

function deformed_variance(delta1, delta2, delta3, delta4)
    println(typeof(delta1))
    dlinks = [[links[i][j] .+ im*delta1 * (j==1 ? 1 : 0) .+
               im*delta2 * (j==2 ? 1 : 0) .+
               im*delta3 * (j==3 ? 1 : 0) .+
               im*delta4 * (j==4 ? 1 : 0) for j in 1:4] for i in 1:10000]
    # push!(logg, dlinks)
    Q = real.([exp.(- W1link_action(dlinks[i], 5.555) .+ W1link_action(links[i], 5.555)) .* exp.(im*W1link(dlinks[i])) for i in 1:10000])
    return var(Q)
end

res = gradient(deformed_variance, ([delta1], [delta2], [delta3], [delta4]))
delta1 -= 0.1 * res[1][1]
delta2 -= 0.1 * res[2][1]
delta3 -= 0.1 * res[3][1]
delta4 -= 0.1 * res[4][1]

delta1 + delta2 - delta3 - delta4





# Working version with ForwardDiff using angle representation


deltas = zeros(4)

links = [angle.([ei.U[1,1,1],ei.U[2,1,2],ei.U[1,2,1],ei.U[1,1,2]]) for ei in ens]

function W1link(lplaqs)
    return lplaqs[1] + lplaqs[2] - lplaqs[3] - lplaqs[4]
end

function W1link_action(lplaqs, beta)
    theta = W1link(lplaqs)
    return -beta*cos.(theta)
end

function deformed_variance(deltas)
    println(typeof(delta1))
    dlinks = [links[i] .+ im*deltas for i in 1:10000]
    Q = real.([exp(- W1link_action(dlinks[i], 5.555) + W1link_action(links[i], 5.555)) * exp(im*W1link(dlinks[i])) for i in 1:10000])
    @show mean(Q)
    return var(Q)
end

res = ForwardDiff.gradient(deformed_variance, deltas)

deltas .-= 0.1 * res

deltas[1]*4

deltas = [0.05, 0.05, 0.05, 0.05]

deformed_variance(deltas)


# Working version with ForwardDiff using link representation


deltas = zeros(4)

links = [[ei.U[1,1,1],ei.U[2,1,2],ei.U[1,2,1],ei.U[1,1,2]] for ei in ens]

function W1link(lplaqs)
    # return lplaqs[1] * lplaqs[2] / lplaqs[3] / lplaqs[4]
    return lplaqs[1] * lplaqs[2] * conj(lplaqs[3]) * conj(lplaqs[4])
end

function W1link_action(lplaqs, beta)
    theta = W1link(lplaqs)
    return -beta*cos(-im*log(theta))
end

function deformed_variance(deltas)
    println(typeof(delta1))
    dlinks = [links[i] .* exp.(-deltas) for i in 1:10000]
    Q = real.([exp(- W1link_action(dlinks[i], 5.555) + W1link_action(links[i], 5.555)) * W1link(dlinks[i]) for i in 1:10000])
    # @show S_def = W1link_action(dlinks[1], 5.555)+5.555
    # @show S_undef = W1link_action(links[1], 5.555)+5.555
    # @show obs = W1link(dlinks[1])
    # @show Q[1]
    @show mean(Q)
    return var(Q)
end

deltas = [0.0, 0.0, 0.0, 0.0]

res = ForwardDiff.gradient(deformed_variance, deltas)
deltas .-= 0.1 * res


deltas = [0.05, 0.05, -0.05, -0.05]

deltas = [0.2, 0.0, 0.0, 0.0]

deformed_variance(deltas)

deltas[1]*4



## Comparison 

deltas = [0.05, 0.05, -0.05, -0.05]

deltas = [0.2, 0.0, 0.0, 0.0]

links_a = [angle.([ei.U[1,1,1],ei.U[2,1,2],ei.U[1,2,1],ei.U[1,1,2]]) for ei in ens]
dlinks_a =  [links_a[i] .+ im*deltas for i in 1:10000]

links_l = [[ei.U[1,1,1],ei.U[2,1,2],ei.U[1,2,1],ei.U[1,1,2]] for ei in ens]
dlinks_l = [links_l[i] .* exp.(-deltas) for i in 1:10000]

links_a == [angle.(links_li) for links_li in links_l]

exp.(im*links_a[1]) - links_l[1]

exp.(im*dlinks_a[1]) - dlinks_l[1]


function W1link_a(lplaqs)
    return lplaqs[1] + lplaqs[2] - lplaqs[3] - lplaqs[4]
end

function W1link_l(lplaqs)
    return lplaqs[1] * lplaqs[2] / lplaqs[3] / lplaqs[4]
end


exp(im*W1link_a(dlinks_a[1])) -  W1link_l(dlinks_l[1])



function W1link_a_action(lplaqs, beta)
    theta = W1link_a(lplaqs)
    return -beta*cos.(theta)
end

function W1link_l_action(lplaqs, beta)
    theta = W1link_l(lplaqs)
    return -beta*real.(theta)
end

function W1link_l_action_good(lplaqs, beta)
    theta = W1link_l(lplaqs)
    return -beta*cos(-im*log(theta))
end

W1link_a_action(dlinks_a[1], 5.555)

W1link_l_action(dlinks_l[1], 5.555)


cos(W1link_a(dlinks_a[1]))

W1link_l(dlinks_l[1])


exp(im*W1link_a(dlinks_a[1]))

W1link_l(dlinks_l[1])


# Taking the real part of exp(i*ϕ) is not the same as cos(ϕ)

real(exp(im*W1link_a(dlinks_a[1])))

cos(W1link_a(dlinks_a[1]))

cos(-im*log(exp(im*W1link_a(dlinks_a[1]))))


W1link_l_action_good(dlinks_l[1],5.555)+5.555








links = [[ei.U[1,1,1],ei.U[2,1,2],ei.U[1,2,1],ei.U[1,1,2]] for ei in ens]

function W1link(lplaqs)
    return lplaqs[1] * lplaqs[2] * conj(lplaqs[3]) * conj( lplaqs[4])
end

function W1link_action(lplaqs, beta)
    theta = W1link(lplaqs)
    return -beta*real.(theta)
end

function deformed_variance(delta1, delta2, delta3, delta4)
    println(typeof(delta1))
    dlinks = [[links[i][j] .* exp.(-delta1) .* (j==1 ? 1 : 0) .*
               exp.(-delta2) .* (j==2 ? 1 : 0) .*
               exp.(-delta3) .* (j==3 ? 1 : 0) .*
               exp.(-delta4) .* (j==4 ? 1 : 0) for j in 1:4] for i in 1:10000]
    # push!(logg, dlinks)
    Q = real.([exp.(- W1link_action(dlinks[i], 5.555) .+ W1link_action(links[i], 5.555)) .* W1link(dlinks[i]) for i in 1:10000])
    return var(Q)
end

res = gradient(deformed_variance, ([delta1], [delta2], [delta3], [delta4]))

delta1 -= 0.1 * res[1][1]
delta2 -= 0.1 * res[2][1]
delta3 -= 0.1 * res[3][1]
delta4 -= 0.1 * res[4][1]

delta1 + delta2 - delta3 - delta4






















# OBC

using ForwardDiff

function W1(model)
    U = model.U
    # return cos(-im*log(U[1,1,1] * U[2,1,2] / U[1,2,1] / U[1,1,2]))
    return U[1,1,1] * U[2,1,2] * conj( U[1,2,1]) *conj( U[1,1,2])
end

stor = []

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
                       BC = OpenBC
                      )
        S_undef = action(Umodel)
        Umodel.U .*= exp.(-delta)
        S_def = action(Umodel)
        obs = W1(Umodel)
        S[i] = exp(-S_def + S_undef) * obs
    end
    push!(stor, S)
    println(mean(S))
    return var(real.(S))
end

compute_vars_comp(x) = compute_var(ens, x)

delta = zeros(Float64, size(model.U))

delta0 = zeros(Float64, size(model.U))

ForwardDiff.gradient(x -> compute_var(ens, x), delta)

delta[1,1,1] = 0.1

ders = ForwardDiff.gradient(compute_vars_comp, delta)

delta .-= 0.1 * ders
# delta[1,1,1] -= 0.1 * ders[1,1,1]
# ders

deltas

delta[1,1,1] = 0.2

compute_vars_comp(delta)

1-mean(gauge_action.(ens))/beta

action(ens[1])

dmod = deepcopy(ens[1])

dmod.U[1,1,1] *= exp(-0.2)

action(dmod)

using ADerrors

W1s = W1.(ens)

W1s = stor[1]


id = "test"

uwW1 = uwreal(real.(W1s), id)
uwerr(uwW1)
uwW1

taui(uwW1, id)


# PBC 


fname = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTExperiments/U1/U1ContourDeformation.jl/run-U1-pbc-b5.0-L2-nc10000.bdio"


beta = 5.0
lsize = 2
model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = PeriodicBC,
                  )

ens = [deepcopy(model) for i in 1:10000]

read_ensemble!(ens, fname)

mean(W1.(ens))

function W1(model)
    U = model.U
    # return cos(-im*log(U[1,1,1] * U[2,1,2] / U[1,2,1] / U[1,1,2]))
    return U[1,1,1] * U[2,1,2] * conj( U[1,2,1]) *conj( U[1,1,2])
end

stor = []

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
        obs = W1(Umodel)
        S[i] = exp(-S_def + S_undef) * obs
    end
    println(mean(S))
    push!(stor, S)
    return var(real.(S))
end

compute_vars_comp(x) = compute_var(ens, x)

delta = zeros(Float64, size(model.U))

mask = zeros(Float64, size(model.U))

mask[1,1,1] = 1.0
mask[1,2,1] = 1.0
mask[1,1,2] = 1.0
mask[2,1,2] = 1.0


ders = ForwardDiff.gradient(compute_vars_comp, delta)
delta .-= 0.1*ders .* mask

delta0 = zeros(Float64, size(model.U))

compute_vars_comp(delta)

using ADerrors

W1s = W1.(ens)

W1s = stor[1]


id = "test"

uwW1 = uwreal(real.(W1s), id)
uwerr(uwW1)
uwW1

taui(uwW1, id)
