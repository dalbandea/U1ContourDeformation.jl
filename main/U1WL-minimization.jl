using Revise
import Pkg
Pkg.activate(".")
using LFTU1, U1ContourDeformation
using BDIO
using Statistics
using FormalSeries

beta = 5.555
lsize = 64

f_plaq = "plaqs-b5.555-L64-nc10000.bdio"

Ps = Vector{Float64}(undef, 10000)

fb = BDIO_open(f_plaq, "r")
BDIO_seek!(fb)
BDIO_read(fb, Ps)


using ADerrors

ID = "test"

uwW1 = uwreal(cos.(Ps), ID)
uwerr(uwW1)
uwW1

std(cos.(Ps))/sqrt(10000)

# Deformed Wilson Loop with FormalSeries

delta = Series{ComplexF64, 2}((0.0, 1.0))

Q = Vector{Series{ComplexF64, 2}}(undef, 10000)

for i in eachindex(Ps)
    Q[i] = real(exp(beta*(cos(Ps[i] - im*delta)-cos(Ps[i])))*exp(im*Ps[i] + delta))
end

var(Q)

uwdW1 = uwreal(dPs, ID)
uwerr(uwdW1)
uwdW1



delta = Series{ComplexF64, 2}((-0.204, 1.0))
Q = Vector{Series{ComplexF64, 2}}(undef, 10000)

for i in eachindex(Ps)
    Q[i] = real(exp(beta*(cos(Ps[i] - im*delta)-cos(Ps[i])))*exp(im*Ps[i] + delta))
end
delta -= my_var(Q)[2]


# With links

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

fname = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/run-b5.555-L64-nc10000.bdio"

links = Vector{Vector{Float64}}(undef, 10000)

LFTU1.randomize!(model)

lplaq(model, 10, 10)

read_observable!(links, x -> lplaq(x, 20, 20), model, fname)

function save_W1links(filename, links)
    fb = BDIO_open(filename, "w", "Links U1")
    for i in 1:length(links)
        BDIO_start_record!(fb, BDIO_BIN_F64LE, 1, true)
        BDIO_write!(fb, links[i])
        BDIO_write_hash!(fb)
    end
    BDIO_close!(fb)
    return nothing
end

function read_W1links(filename, links)
    fb = BDIO_open(filename, "r")
    cont = 1
    while BDIO_seek!(fb)
        if BDIO_get_uinfo(fb) == 1
            BDIO_read(fb, links[cont])
            cont += 1
        end
    end
    BDIO_close!(fb)
    return nothing
end

save_W1links("linkstest.bdio", links)

links2 = [Vector{Float64}(undef, 4) for i in 1:10000]

read_W1links("linkstest.bdio", links2)

links = links2

function W1link(lplaqs)
    return lplaqs[1] + lplaqs[2] - lplaqs[3] - lplaqs[4]
end

function W1link_action(lplaqs, beta)
    theta = W1link(lplaqs)
    return -beta*cos(theta)
end

mean(cos.(W1link.(links)))

dlinks = [[links[i][j] + im*0.2 * (j==1 ? 1 : 0) for j in 1:4] for i in 1:10000]

Q = real.([exp(- W1link_action(dlinks[i], 5.555) + W1link_action(links[i], 5.555)) * exp(im*W1link(dlinks[i])) for i in 1:10000])

mean(Q)

links

# Deforming all links

using ReverseDiff
import ReverseDiff: gradient

delta1 = 0.0
delta2 = 0.0
delta3 = 0.0
delta4 = 0.0

deltas = (delta1, delta2, delta3, delta4)

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
    push!(logg, dlinks)
    Q = real.([exp.(- W1link_action(dlinks[i], 5.555) .+ W1link_action(links[i], 5.555)) .* exp.(im*W1link(dlinks[i])) for i in 1:10000])
    return var(Q)
end

res = gradient(deformed_variance, ([delta1], [delta2], [delta3], [delta4]))

delta1 -= 0.1 * res[1][1]
delta2 -= 0.1 * res[2][1]
delta3 -= 0.1 * res[3][1]
delta4 -= 0.1 * res[4][1]

delta1 + delta2 - delta3 - delta4

gradient(f, ([2.0], [2.0]))

f(x, y) = x.^2 + y.^2


# Deforming with self-defined structs

import ReverseDiff
import ReverseDiff.gradient

beta = 5.555
lsize = 64

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

function daction(model, delta)
    U = copy(model.U) .* exp.(im*delta) .* exp.(-im*delta)
    model2 = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize, lsize),
                       BC = OpenBC,
                       custom_init = U
                      )
    model2.U[1,1,1] *= exp(im*delta[1])
    return action(model2)
end

@time gradient(x -> daction(model, x), [1.0])

const f_tape = ReverseDiff.GradientTape(x -> daction(model, x), [1.0])

const compiled_f_tape = ReverseDiff.compile(f_tape)

result = [0.0]

@time ReverseDiff.gradient!(result, compiled_f_tape, [1.0])

logg[1]

logg[2]



# model.U[1,1,1] *= exp(im*1.0)
a1 = action(model)
model.U[1,1,1] *= exp(im*0.000001)
a2 = action(model)
(a2-a1)/0.0001


fname = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/run-b5.555-L64-nc10000.bdio"

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

ens2 = ens[1:100]

function W1(model, i1, i2)
    iu1 = mod1(i1+1, model.params.iL[1])
    iu2 = mod1(i2+1, model.params.iL[2])
    return model.U[i1,i2,1] * model.U[iu1,i2,2] * conj(model.U[i1,iu2,1] * model.U[i1,i2,2])
end

function def_1link!(model::LFTU1.U1, delta)
    model.U[1,1,1] *= exp(im*delta[1])
    return nothing
end

function compute_var(ensemble, delta)
    Q = Vector{Any}(undef, 10000)
    for i in eachindex(ensemble)
        U = copy(ens[i].U) .* exp.(im*delta) .* exp.(-im*delta)
        println(i)
        # println("A")
        model = U1Quenched(Float64,
                           # FormalSeries.Series{ComplexF64,2},
                           beta = ens[i].params.beta,
                           iL = (ens[i].params.iL[1], ens[i].params.iL[1]),
                           BC = ens[i].params.BC,
                           custom_init = U
                          )
        # model.U .= ens[1].U
        # println("B")
        S_undef = action(model)
        def_1link!(model, delta)
        # println("C")
        S_def = action(model)
        # println("D")
        obs = W1(model, 1, 1)
        Q[i] = exp(-S_undef + S_def) * obs
    end
    println(mean(Q))
    return var(real.(Q))
end

using ForwardDiff

ForwardDiff.gradient(x -> compute_var(ens, x), [-0.2])

compute_var(ens, -0.2)

using Zygote

Zygote.gradient(x -> compute_var(ens, x), 0.1)


const f_tape = ReverseDiff.GradientTape(x -> compute_var(ens2, x), [0.0])
const compiled_f_tape = ReverseDiff.compile(f_tape)

result = [0.0]

@time ReverseDiff.gradient!(result, f_tape, [1.0])
