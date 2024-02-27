install.packages("devtools")
devtools::install_github("QizhiSu/mspcompiler", build_vignettes = TRUE)

library(mspcompiler)
library(future)
library(future.apply)
library(parallel)
plan(multisession(workers = detectCores() - 1))

nist_ei <- read_lib("NIST.MSP", type = "EI")
riken_ei <- read_lib("GCMS DB-Public-KovatsRI-VS3.msp",
                     type = "EI", remove_ri = FALSE)
mona_ei <- read_lib("MoNA-export-GC-MS_Spectra.msp", type = "EI")
mona_ei <- reorganize_mona(mona_ei)

combine_ei <- c(nist_ei, riken_ei, mona_ei)
nist_ri <- extract_ri("ri.dat", "USER.DBU")
combine_ei <- assign_ri(combine_ei, nist_ri, polarity = "semi-polar")
plan(sequential)
