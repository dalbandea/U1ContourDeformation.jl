using Revise
import Pkg
Pkg.activate(".")
using LFTU1, U1ContourDeformation
using BDIO

function plaq(model::LFTU1.U1, i1, i2)
    U = model.U
    iu1 = mod1(i1+1, model.params.iL[1])
    iu2 = mod1(i2+1, model.params.iL[2])
    return angle(U[i1,i2,1] * U[iu1,i2,2] * conj(U[i1,iu2,1] * U[i1,i2,2]))
end

beta = 5.555
lsize = 64

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

fname = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/run-b5.555-L64-nc10000.bdio"

fb = BDIO_open(fname, "r")

Ss = Vector{Float64}()

Ps = Vector{Float64}()

cont = 0
while BDIO_seek!(fb)
    if BDIO_get_uinfo(fb) == 1
        BDIO_read(fb, model)
        # push!(Ss, action(model))
        push!(Ps, plaq(model, 20, 20))
        cont += 1
        println(cont)
    end
end

BDIO_close!(fb)

f_plaq = "plaqs-b5.555-L64-nc10000.bdio"

fb = BDIO_open(f_plaq, "w", "Plaquettes U1")
BDIO_start_record!(fb, BDIO_BIN_F64LE, 1, true)
BDIO_write!(fb, Ps)
BDIO_close!(fb)

fb = BDIO_open(f_plaq, "r")
BDIO_seek!(fb)
testps = similar(Ps)
BDIO_read(fb, testps)

using ADerrors

ID = "test"

uwS = uwreal(Ss, ID)

uwP = 1 - uwS/(model.params.beta*(model.params.iL[1]-1)^2)
uwerr(uwP)
uwP

uwW = -log(uwP)
uwerr(uwW)
uwW

# Undeformed Wilson Loop

uwW1 = uwreal(cos.(Ps), ID)
uwerr(uwW1)
uwW1

# Deformed Wilson Loop

delta = -0.204

dPs = @. real(exp(beta*(cos(Ps - im*delta)-cos(Ps)))*exp(im*Ps + delta))

uwdW1 = uwreal(dPs, ID)
uwerr(uwdW1)
uwdW1

# Is this equivalent to reweighting?

delta = 0.112
shPs = @. Ps + im*delta
w = @. exp(beta * (cos(shPs) - cos(shPs - im*delta)))
num = uwreal(real.(exp.(im*shPs).*w), ID)
den = uwreal(real(w), ID)
rwW1 = num / den
uwerr(rwW1)
rwW1


# Test read all ensemble

fname = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/run-b5.555-L64-nc10000.bdio"

beta = 5.555
lsize = 64
model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

ens = [deepcopy(model) for i in 1:10000]

read_ensemble!(ens, fname)

Ps = zeros(10000)

for i in 1:10000
    # Ps[i] = plaq(ens[i], 20, 20)
    Ps[i] = action(ens[i])
    println(i)
end


using FormalSeries

beta = 5.555
lsize = 64

model = U1Quenched(Float64,
                   Series{ComplexF64, 2},
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )
