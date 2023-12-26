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

fname = "/home/david/scratch/projects/phd/12-Contour-deformation/run-TU1Nf2-b5.555-m0.6-L2-BCPeriodicBC-nc10000.bdio"
