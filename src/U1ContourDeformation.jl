module U1ContourDeformation

import LFTSampling
import LFTU1
import BDIO
import ADerrors
import Statistics
import FormalSeries
import ReverseDiff

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
        if BDIO.BDIO_get_uinfo(fb) == 1
            LFTU1.BDIO_read(fb, ens[cont], reg)
            println(cont)
            cont += 1
        end
    end
    BDIO.BDIO_close!(fb)
    return nothing
end
export read_ensemble!

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
