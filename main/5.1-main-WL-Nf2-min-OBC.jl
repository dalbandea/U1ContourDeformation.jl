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

# -L 2 -b 5.555 --mass 0.6 --BC OpenBC --nsteps 7 -n 10000 --wdir /home/david/scratch/projects/phd/12-Contour-deformation/ensembles/
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-A"
        help = "Wilson Loop area"
        required = true
        arg_type = Int

        "--enspath"
        help = "path of ensemble to analyze"
        required = true
        arg_type = String
        # default = "configs/"
        
        "--wdir"
        help = "path to directory to save configurations and logs"
        required = true
        arg_type = String
    end

    return parse_args(s)
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

using LinearAlgebra
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
            D[i,j] = dot(x2, xtmp)
            xtmp[j] = 0.0
        end
        x1[i] = 0.0
    end
end

function compute_var(ensemble, delta, mask, A)
    deltas = build_minimal_deltas(delta, mask, A)
    S = Vector{Any}(undef, length(ensemble))
    x1 = similar(copy(ensemble[1].U) .* exp.(im*deltas) .* exp.(-im*deltas))
    x2 = similar(x1)
    xtmp = similar(x1)
    D = similar(x1, prod(size(x1)), prod(size(x1)))
    U = copy(ensemble[1].U) .* exp.(im*deltas) .* exp.(-im*deltas)
    Umodel = instance_from_config(ens[1], U)
    for i in eachindex(ensemble)
        # println(i)
        Umodel.U .= ensemble[i].U
        S_undef = complex_action(Umodel)
        compute_Dwsr!(D, x1, x2, xtmp, Umodel)
        # maxD_undef = maximum(real.(D))
        detD_undef = det(D)
        Umodel.U .*= exp.(-deltas)
        S_def = complex_action(Umodel)
        obs = W1(Umodel)
        compute_Dwsr!(D, x1, x2, xtmp, Umodel)
        # maxD_def = maximum(real.(D))
        detD_def = det(D)
        S[i] = exp(-S_def + S_undef) #=* (maxD_def/maxD_undef)^(prod(size(Umodel.U)))=# * detD_def / detD_undef * obs
    end
    # println("hi")
    # push!(stor, S)
    # println(mean(S))
    # println(real.(S))
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

########################
# PARSE ARGUMENTS ######
########################

parsed_args = parse_commandline()

A       = parsed_args["A"]
fname   = parsed_args["enspath"]
wdir    = parsed_args["wdir"]

wdir = joinpath(wdir, splitpath(fname)[end], "A$A")
mkpath(wdir)

fb, model = LFTU1.read_cnfg_info(fname, U1Nf2)
ens = [deepcopy(model) for i in 1:10000]
read_ensemble!(ens, fname)

mask = zeros(Float64, size(ens[1].U))
delta_wloop_mask!(mask, A)
delta = zeros(length(delta_wloop_indices(A)))
ders = zeros(length(delta_wloop_indices(A)))
res = Results()
update_results!(res, real.([wilson_loop(A, ens[i])  for i in 1001:length(ens)]), delta, ders)

for i in 1:100
    print(i,"\r")
    ders .= compute_ders_obs(ens[1:10:1000], delta, mask, A)
    delta .-= (i > 10 ? 1.0 : 0.1) * ders
    _, defobs = compute_var(ens[1001:end], delta, mask, A)
    update_results!(res, real.(defobs), delta, ders)
end

fileobs = joinpath(wdir, "obs.txt")
filedelta = joinpath(wdir, "deltas.txt")
fileders = joinpath(wdir, "ders.txt")
filemdata = joinpath(wdir, "metadata.txt")

writedlm(fileobs, hcat(res.means, res.errs, res.vrncs, res.taus, res.dtaus), ',')
writedlm(filedelta, res.delts, ',')
writedlm(fileders, res.ders, ',')

writedlm(filemdata, "fname = $fname\nA = $A") 
