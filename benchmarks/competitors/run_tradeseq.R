# tradeSeq conditionTest on liver hepatocytes (2,000 per donor, 14,000 cells; prep_liver.py, seed 0).
# Pseudotime = capped distance to cholangiocyte on [0, 1]; one lineage, cellWeights = 1; conditions = split groups.
# tradeSeq treats cells as replicates (no donor term).
# Usage: Rscript run_tradeseq.R <split>
suppressPackageStartupMessages({ library(tradeSeq); library(Matrix); library(BiocParallel) })
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
res <- file.path(here, "results"); split <- commandArgs(TRUE)[1]
cells <- read.csv(file.path(res, "sub_cells.csv"))
counts <- as(readMM(file.path(res, "sub_counts.mtx")), "CsparseMatrix")
rownames(counts) <- readLines(file.path(res, "genes.txt")); colnames(counts) <- cells$cell
sp <- read.csv(file.path(res, "splits.csv")); sp <- sp[sp$split == split, ]
cond <- factor(sp$group[match(cells$donor, sp$donor)])
pt <- matrix(cells$dist, ncol = 1); cw <- matrix(1, nrow(cells), 1)
set.seed(0)
t0 <- Sys.time()
sce <- fitGAM(counts = counts, pseudotime = pt, cellWeights = cw, conditions = cond, nknots = 6,
              verbose = FALSE, parallel = FALSE)
ct <- conditionTest(sce)
ct$gene <- rownames(ct); ct$FDR <- p.adjust(ct$pvalue, "BH")
write.csv(ct, file.path(res, sprintf("tradeseq_%s.csv", split)), row.names = FALSE)
ok <- !is.na(ct$pvalue)
cat(sprintf("tradeseq %s: %d genes tested, FDR<.05 %d, p<.05 %.3f, %.1f min\n", split, sum(ok),
            sum(ct$FDR < .05, na.rm = TRUE), mean(ct$pvalue[ok] < .05), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
