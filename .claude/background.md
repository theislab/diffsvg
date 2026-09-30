# Introduction

This document summarizes the purpose behind this package.

# Scientific Background

In spatial transcriptomics it is possible to analyze variables e.g. gene counts w.r.t. morphological markers, i.e. you can track how expression changes with increasing distance to an "anchor point" (which can be coordinate based or a cluster of cells or some annotated region). Comparing this expression trend between samples or conditions is where the differential aspect comes in. We want to detect genes (named spatial differentially expressed genes or differential spatially variable genes) which have different expression patterns by distance between groups.

# Method

To detect these genes we run a generalized additive model defined using PyMC and rank the genes using a wald statistic on interaction term coefficients or by calculating an empirical p-value if the panel size is large enough.

# Data

Currently we mainly use a Xenium Mouse brain dataset with the dentate gyrus used as anchor.

# Open problems

A precise hypothesis test is not trivial, as most genes will have slight differences between conditions, causing the test statistic to quickly get huge, resulting in p-value inflation. The temporary fix is to use the wald statistic for ranking alone and separately calculating empirical p-values. The p-value distribution is fine now, but depends on the gene panel. So you can't get a meaningful p-value when only probing one gene, because you need a null distribution to compare to. Additionally we use binned data because most spatial data consists of zero counts which add little information but slow down the model fitting process drastically.
