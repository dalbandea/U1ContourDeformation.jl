module U1ContourDeformation

using LFTSampling
using LFTU1
import BDIO
import ADerrors
import Statistics
import FormalSeries
import ReverseDiff
import Random
import SpecialFunctions: besseli
import LinearAlgebra

import Statistics: var
"""
    var(x::Vector{FormalSeries.Series{T,N}})

Modification of Statistics.var to compute the variance of a vector of Series
"""
function Statistics.var(x::Vector{FormalSeries.Series{T,N}}) where {T,N}
    mx = Statistics.mean(x)
    v = zero(typeof(mx))
    for i in eachindex(x)
        v+= (x[i] - mx)^2 / (length(x)-1)
    end
    return v
end
export var

"""
    plaq(model::LFTU1.U1, i1, i2)

returns angle of plaquette at position (i1, i2)
"""
function plaq(model::LFTU1.U1, i1, i2)
    U = model.U
    iu1 = mod1(i1+1, model.params.iL[1])
    iu2 = mod1(i2+1, model.params.iL[2])
    return angle(U[i1,i2,1] * U[iu1,i2,2] * conj(U[i1,iu2,1] * U[i1,i2,2]))
end
export plaq

"""
    lplaq(model::LFTU1.U1, i1, i2)

returns angle of links which build up the plaquette at (i1, i2)
"""
function lplaq(model::LFTU1.U1, i1, i2)
    U = model.U
    iu1 = mod1(i1+1, model.params.iL[1])
    iu2 = mod1(i2+1, model.params.iL[2])
    return [angle(U[i1,i2,1]), angle(U[iu1,i2,2]), angle(U[i1,iu2,1]), angle(U[i1,i2,2])]
end
export lplaq


"""
    read_observable!(obs, op, fname)

operator `op` is applied on each configuration of the BDIO file `fname` and is
read into vector `obs` (needs to have same number of elements as elements to be
read)
"""
function read_observable!(obs, op, lftws, fname)
    fb = BDIO.BDIO_open(fname, "r")
    cont = 1
    while BDIO.BDIO_seek!(fb)
        if BDIO.BDIO_get_uinfo(fb) == 1
            LFTU1.BDIO_read(fb, lftws)
            obs[cont] = op(lftws)
            println(cont)
            cont += 1
        end
    end
    BDIO.BDIO_close!(fb)
    return nothing
end
export read_observable!

function read_ensemble!(ens, fname)
    fb = BDIO.BDIO_open(fname, "r")
    cont = 1
    reg = zeros(Float64, 2)
    while BDIO.BDIO_seek!(fb)
        if BDIO.BDIO_get_uinfo(fb) == 8
            LFTU1.BDIO_read(fb, ens[cont], reg)
            println(cont)
            cont += 1
        end
    end
    BDIO.BDIO_close!(fb)
    return nothing
end
export read_ensemble!


function complex_action(u1ws::LFTU1.U1)

    Nx = u1ws.params.iL[1]
    Ny = u1ws.params.iL[2]
    U = u1ws.U

    res = zero(eltype(u1ws.U))

    for i1 in 1:Nx-1*(u1ws.params.BC==OpenBC ? 1 : 0), i2 in 1:Ny-1*(u1ws.params.BC==OpenBC ? 1 : 0)
        iu1 = mod(i1, Nx) + 1
        iu2 = mod(i2, Ny) + 1

        res += cos(-im*log(U[i1,i2,1] * U[iu1,i2,2] / U[i1,iu2,1] / U[i1,i2,2]))
        # println(i1)
    end

    if u1ws.params.BC == OpenBC
        Nx = Nx - 1
        Ny = Ny - 1
    end

    return u1ws.params.beta * (Nx * Ny - res)
end
export complex_action

function W1(model)
    U = model.U
    return U[1,1,1] * U[2,1,2] / U[1,2,1] / U[1,1,2]
end
export W1

function instance_from_config(model::LFTU1.U1Quenched, U)
    modeltype = supertype(typeof(model))
    return modeltype(Float64,
              custom_init = U,
              beta = model.params.beta,
              iL = (model.params.iL[1], model.params.iL[2]),
              BC = model.params.BC
              )
end

function instance_from_config(model::LFTU1.U1Nf2, U)
    modeltype = supertype(typeof(model))
    return modeltype(
                      Float64,
                      custom_init = U,
                      beta = model.params.beta,
                      iL = (model.params.iL[1], model.params.iL[2]),
                      am0 = model.params.am0,
                      BC = model.params.BC
                    )
end
export instance_from_config

import ADerrors: uwreal
function uwreal(data::Vector{Float64})
    id = Random.randstring(12)
    uwx = uwreal(data, id)
    return uwx, id
end
function ana_obs(x)
    uwx, id = uwreal(x)
    ADerrors.uwerr(uwx)
    tau = ADerrors.taui(uwx, id)
    dtau = ADerrors.dtaui(uwx, id)
    println("Var = ", var(x))
    println("Mean = ", uwx)
    println("tau = ", tau, " +/- ", dtau)
    return nothing
end
export ana_obs



function wilson_plaquette(u1ws, i1=1, i2=1)
    Nx = u1ws.params.iL[1]
    Ny = u1ws.params.iL[2]
    U = u1ws.U
    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1
    return U[i1,i2,1] * U[iu1,i2,2] / U[i1,iu2,1] / U[i1,i2,2]
end
export wilson_plaquette

function wilson_loop(A, u1ws::LFTU1.U1)
    wlsize = convert(Int64, floor(sqrt(A)))
    rmdr = A - wlsize^2
    contA = zero(eltype(A))
    res = one(eltype(u1ws.U))
    for i1 in 1:wlsize, i2 in 1:wlsize
        # println("($i1, $i2)")
        res = res * wilson_plaquette(u1ws, i1, i2)
    end
    if rmdr == 0
        return res
    end
    for i2 in 1:min(rmdr, wlsize+1)
        # println("($(wlsize+1), $i2)")
        res = res * wilson_plaquette(u1ws, wlsize+1, i2)
    end
    if rmdr > wlsize+1
        for i1 in wlsize:-1:wlsize-(rmdr-(wlsize+1))+1
            # println("($i1, $(wlsize+1))")
            res = res * wilson_plaquette(u1ws, i1, wlsize+1)
        end
    end
    return res
end
export wilson_loop


"""
    wilson_loop_indices(A)

returns positions of Wilson plaquettes which make up a Wilson loop of size `A`
"""
function wilson_loop_indices(A)
    indices = []
    wlsize = convert(Int64, floor(sqrt(A)))
    rmdr = A - wlsize^2
    contA = zero(eltype(A))
    res = zero(Float64)
    for i1 in 1:wlsize, i2 in 1:wlsize
        push!(indices, (i1, i2))
    end
    if rmdr == 0
        return indices
    end
    for i2 in 1:min(rmdr, wlsize+1)
        push!(indices, (wlsize+1, i2))
    end
    if rmdr > wlsize+1
        for i1 in wlsize:-1:wlsize-(rmdr-(wlsize+1))+1
            push!(indices, (i1, wlsize+1))
        end
    end
    return indices
end
export wilson_loop_indices

"""
    delta_wloop_indices(A)

returns link coordinates of all links inside a Wilson loop of size `A`
"""
function delta_wloop_indices(A)
    plaq_indices = wilson_loop_indices(A)
    indices = []
    for plaq_index in plaq_indices
        i1 = convert(Int64, plaq_index[1])
        i2 = convert(Int64, plaq_index[2])
        push!(indices, (i1, i2, 1))
        push!(indices, (i1+1, i2, 2))
        push!(indices, (i1, i2+1, 1))
        push!(indices, (i1, i2, 2))
    end
    unique!(indices)
    return indices
end
export delta_wloop_indices

"""
    delta_wloop_mask!(mask, A)

overwrites `mask` with 1 on coordinates of all links inside a Wilson loop of size
`A`, and 0 elsewhere
"""
function delta_wloop_mask!(mask, A)
    floor(sqrt(A))+1 >= size(mask, 1) && error("mask size too small")
    indices = delta_wloop_indices(A)
    mask .= 0
    for index in indices
        mask[index...] = 1
    end
end
export delta_wloop_mask!

"""
    build_minimal_deltas(deltas, mask, A)

returns matrix with the shape of `mask` with the `delta` values stored in the
indices as returned by `delta_wloop_indices(A)`. Useful so that ForwardDiff only
computes the derivatives of the elements in `deltas`.
"""
function build_minimal_deltas(deltas, mask, A)
    idcs = delta_wloop_indices(A)

    deltas_min = zeros(eltype(deltas),size(mask))

    for i in eachindex(deltas)
        deltas_min[idcs[i]...] = deltas[i]
    end

    return deltas_min
end
export build_minimal_deltas


"""
    builtup_minimal_delta!(mask, A, delta, delta0 = 0; dims = 1)

returns cooked delta field that have A plaquettes with the given value of `delta`
"""
function builtup_minimal_delta!(mask, A, delta, delta0 = 0; dims = 1)
    indices = delta_wloop_indices(A)
    mask .= 0.0

    for index in indices
        if index[end] == 2
            mask[index...] = + delta0 + (index[1]-1) * delta
        end
        if dims == 2
            if index[end] == 1
                mask[index...] = -delta0 - (index[2]-1) * delta
            end
            mask[index...] *= 0.5
        end
    end
    return nothing
end
export builtup_minimal_delta!



function ana_wilson_loop(A, beta)
    return (besseli(1, beta)/besseli(0, beta))^A
end
export ana_wilson_loop

function compute_Dwsr!(D, x1, x2, xtmp, xmodel)
    D .= 0.0
    x1 .= 0.0
    x2 .= 0.0
    xtmp .= 0.0
    for i in eachindex(x1)
        x1[i] = 1.0
        gamm5Dw_sqr_msq!(x2, xtmp, x1, xmodel)
        for j in eachindex(xtmp)
            D[j,i] = x2[j]
        end
        x1[i] = 0.0
    end
end
function compute_Dwsr(xmodel)
    x1 = similar(copy(xmodel.U))
    x2 = similar(x1)
    xtmp = similar(x1)
    D = similar(x1, prod(size(x1)), prod(size(x1)))
    compute_Dwsr!(D, x1, x2, xtmp, xmodel)
    return D
end
export compute_Dwsr!, compute_Dwsr

function compute_Dwsr_wog5!(D, x1, x2, xtmp, xmodel)
    D .= 0.0
    x1 .= 0.0
    x2 .= 0.0
    xtmp .= 0.0
    for i in eachindex(x1)
        x1[i] = 1.0
        Dw_sqr_msq!(x2, xtmp, x1, xmodel)
        for j in eachindex(xtmp)
            D[j,i] = x2[j]
        end
        x1[i] = 0.0
    end
end
function compute_Dwsr_wog5(xmodel)
    x1 = similar(copy(xmodel.U))
    x2 = similar(x1)
    xtmp = similar(x1)
    D = similar(x1, prod(size(x1)), prod(size(x1)))
    compute_Dwsr_wog5!(D, x1, x2, xtmp, xmodel)
    return D
end
export compute_Dwsr_wog5!, compute_Dwsr_wog5


function compute_Dwsr_rat!(D, x1, x2, xtmp, sws, defmodel, undefmodel)
    D .= 0.0
    x1 .= 0.0
    x2 .= 0.0
    xtmp .= 0.0
    for i in eachindex(x1)
        x1[i] = 1.0
        gamm5Dw_sqr_msq!(x2, xtmp, x1, defmodel)
        xtmp .= x2
        invc = invert!(x2, gamm5Dw_sqr_msq!, xtmp, sws, undefmodel)
        for j in eachindex(xtmp)
            D[j,i] = x2[j]
        end
        x1[i] = 0.0
    end
end
export compute_Dwsr_rat!

function st_g5Dw_sr_ratio(nsr, Uundef, Udef)
    x1 = similar(eltype(Udef))
    x2 = similar(eltype(Udef))
    xtmp = similar(eltype(Udef))
    sws = CG(10000, 1e-16, Udef.U);
    return st_g5Dw_sr_ratio(nsr, Uundef, Udef, x1, x2, xtmp, sws)
end
function st_g5Dw_sr_ratio(nsr, Uundef, Udef, x1, x2, xtmp, sws)
    eta = randn(ComplexF64, size(Uundef.U))
    res = []
    for i in 1:nsr
        eta .= randn(ComplexF64, size(Uundef.U))
        gamm5Dw_sqr_msq!(x1, xtmp, eta, Udef)
        invert!(x2, gamm5Dw_sqr_msq!, x1, sws, Uundef)
        push!(res, exp(-LinearAlgebra.dot(eta, x2)+LinearAlgebra.dot(eta,eta)))
    end
    return res
end
export st_g5Dw_sr_ratio

# import Base: zero, zeros
# function Base.zero(x::Type{ReverseDiff.TrackedReal{Float64, Float64, Nothing}})
#     return ReverseDiff.track(zero(Float64))
# end

# function Base.zero(x::Type{ReverseDiff.TrackedReal{Float64, Float64, Nothing}})
#     return ReverseDiff.track(zero(Float64))
# end

# function Base.zeros(x::Type{ReverseDiff.TrackedReal{Float64, Float64, Nothing}}, size)
#     X = zeros(Float64, size)
#     X .= zero(x)
#     return X
# end



# import LFTU1: U1quenchedaction
# function U1quenchedaction(U::Array{Complex{ReverseDiff.TrackedReal{Float64, Float64, Nothing}}, 3}, beta, Nx, Ny, BC, device, threads, blocks)
#     plaquettes = real(U[:,:,1])
#     plaquettes .= 0.0
#     plaquettes = LFTU1.to_device(device, plaquettes)
#     println(typeof(plaquettes))
#     return U1quenchedaction(plaquettes, U, beta, Nx, Ny, BC, device, threads, blocks)
# end

end # module U1ContourDeformation
