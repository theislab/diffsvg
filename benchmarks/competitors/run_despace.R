# DESpace2 dsp_test (differential spatial patterns, multi-sample) on liver hepatocytes.
# Adaptation (ours): DESpace needs discrete spatial clusters; we use 5 equal-width bins of the [0, 1]
# capped distance-to-cholangiocyte as the clusters (prep_liver.py).
# Input is one pseudo-cell per donor x bin carrying the summed counts. dsp_test pseudobulks by
# sum over sample x cluster, so this gives the same pseudobulk matrix as passing all 393k cells.
# Its cluster filter (>= 0.5% of cells per condition) was checked on real cell counts: every bin passes.
# Usage: Rscript run_despace.R <split> <design: default|donor>
suppressPackageStartupMessages({
  library(SingleCellExperiment); library(edgeR); library(BiocParallel); library(S4Vectors)
  library(Matrix); library(dplyr); library(data.table); library(scuttle)
})
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "vendor", "DESpace-internal.R")); source(file.path(here, "vendor", "dsp_test.R"))
args <- commandArgs(TRUE); split <- args[1]; dmode <- args[2]
res <- file.path(here, "results")

pb <- read.csv(file.path(res, "counts_pb.csv"), check.names = FALSE, row.names = 1)
genes <- colnames(pb)
sp <- read.csv(file.path(res, "splits.csv")); sp <- sp[sp$split == split, ]
cd <- data.frame(donor = sub("\\|.*", "", rownames(pb)), bin = paste0("b", sub(".*\\|", "", rownames(pb))))
cd$condition <- sp$group[match(cd$donor, sp$donor)]
cd[] <- lapply(cd, factor)  # dsp_test expects factor columns
sce <- SingleCellExperiment(assays = list(counts = t(as.matrix(pb))), colData = DataFrame(cd))
rownames(sce) <- genes; colnames(sce) <- rownames(pb)

design <- NULL
if (dmode == "donor") {
  # Rows must follow .layer_test's pseudobulk column order: expand.grid(sample, cluster), sample fastest.
  md <- expand.grid(donor = levels(factor(cd$donor)), bin = levels(factor(cd$bin)), stringsAsFactors = FALSE)
  md$condition <- factor(sp$group[match(md$donor, sp$donor)])
  X <- model.matrix(~ donor + bin, md)          # donor intercepts absorb the condition main effect
  ref <- levels(md$condition)[1]
  for (b in levels(factor(md$bin))[-1])         # condition x cluster interactions (named with ':')
    X <- cbind(X, setNames(data.frame((md$condition != ref) * (md$bin == b)), paste0("condition:", b)) |> as.matrix())
  rownames(X) <- paste0(md$donor, "_", md$bin)
  design <- X
}
out <- suppressMessages(dsp_test(sce, design = design, cluster_col = "bin", sample_col = "donor",
                                 condition_col = "condition", test = "QLF", verbose = FALSE))
out <- out[match(genes, out$gene_id), ]
write.csv(out, file.path(res, sprintf("despace_%s_%s.csv", dmode, split)), row.names = FALSE)
cat(sprintf("despace %s %s: FDR<.05 %d, p<.05 %.3f\n", dmode, split, sum(out$FDR < .05), mean(out$PValue < .05)))
