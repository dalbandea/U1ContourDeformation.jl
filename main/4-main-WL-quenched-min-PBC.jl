using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using U1ContourDeformation
using Statistics
using ForwardDiff
using ADerrors
using DelimitedFiles

Base.@kwdef mutable struct Results
    means   = Vector{Float64}()
    errs    = Vector{Float64}()
    vrncs   = Vector{Float64}()
    taus    = Vector{Float64}()
    dtaus   = Vector{Float64}()
    delts   = Vector{Vector{Float64}}()
    ders    = Vector{Vector{Float64}}()
end


function ana_obs!(res::Results, x)
    uwx, id = uwreal(x)
    ADerrors.uwerr(uwx)
    mn = ADerrors.value(uwx)
    emn = ADerrors.err(uwx)
    tau = ADerrors.taui(uwx, id)
    dtau = ADerrors.dtaui(uwx, id)
    vr = var(x)
    push!(res.means, mn)
    push!(res.errs, emn)
    push!(res.vrncs, vr)
    push!(res.taus, tau)
    push!(res.dtaus, dtau)
    return nothing
end

function update_results!(res::Results, x, delta, ders)
    ana_obs!(res, x)
    push!(res.delts, copy(delta))
    push!(res.ders, copy(ders))
    return nothing
end

function compute_var(ensemble, delta, mask, A)
    deltas = build_minimal_deltas(delta, mask, A)
    S = Vector{Any}(undef, length(ensemble))
    U = copy(ensemble[1].U) .* exp.(im*deltas) .* exp.(-im*deltas)
    Umodel = instance_from_config(ens[1], U)
    for i in eachindex(ensemble)
        Umodel.U .= ensemble[i].U
        S_undef = complex_action(Umodel)
        Umodel.U .*= exp.(-deltas)
        S_def = complex_action(Umodel)
        obs = wilson_loop(A, Umodel)
        S[i] = exp(-S_def + S_undef) * obs
    end
    return var(real.(S)), S
end

function compute_ders_var(ensemble, deltas, mask, A)
    vrnc, obs = compute_var(ensemble, deltas, mask, A)
    return vrnc
end

function compute_ders_obs(ensemble, deltas, mask, A)
    ders = ForwardDiff.gradient(x -> compute_ders_var(ensemble, x, mask, A), deltas)
    return ders
end

fname = "/home/david/scratch/projects/phd/12-Contour-deformation/ensembles/run-TU1Quenched-b5.555-L24-BCPeriodicBC-nc10000.bdio"
fb, model = LFTU1.read_cnfg_info(fname, U1Quenched)
ens = [deepcopy(model) for i in 1:10000]

read_ensemble!(ens, fname)

for A in 1:16
    mask = zeros(Float64, size(ens[1].U))
    delta_wloop_mask!(mask, A)
    delta = zeros(length(delta_wloop_indices(A)))
    ders = zeros(length(delta_wloop_indices(A)))
    res = Results()
    update_results!(res, real.([wilson_loop(A, ens[i])  for i in 1001:length(ens)]), delta, ders)

    for i in 1:200
        print(i,"\r")
        ders .= compute_ders_obs(ens[1:1000], delta, mask, A)
        delta .-= 0.1 * ders
        _, defobs = compute_var(ens[1001:end], delta, mask, A)
        update_results!(res, real.(defobs), delta, ders)
    end

    wdir = "/home/david/scratch/projects/phd/12-Contour-deformation/analysis/run-TU1Quenched-b5.555-L24-BCPeriodicBC-nc10000/A$A"
    mkpath(wdir)

    fileobs = joinpath(wdir, "obs.txt")
    filedelta = joinpath(wdir, "deltas.txt")
    fileders = joinpath(wdir, "ders.txt")
    filemdata = joinpath(wdir, "metadata.txt")

    writedlm(fileobs, hcat(res.means, res.errs, res.vrncs, res.taus, res.dtaus), ',')
    writedlm(filedelta, res.delts, ',')
    writedlm(fileders, res.ders, ',')

    writedlm(filemdata, "fname = $fname\nA = $A") 
end

