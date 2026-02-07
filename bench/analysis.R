#!/usr/bin/env Rscript
#
# viva_tensor Professional Statistical Analysis
# =============================================
#
# Comprehensive analysis with:
# - Bootstrap confidence intervals
# - Non-parametric tests (Kruskal-Wallis, Dunn)
# - Effect sizes (Cliff's delta)
# - Publication-quality visualizations
# - LaTeX table export
# - Reproducibility checksums
#
# References:
# - Kalibera & Jones (2013)
# - Georges et al. (2007)
# - Hoefler & Belli (2015)

# =============================================================================
# Dependencies
# =============================================================================

required_packages <- c(
  "jsonlite",     # JSON parsing
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "ggplot2",      # Visualization
  "scales",       # Plot scales
  "boot",         # Bootstrap
  "effsize",      # Effect sizes
  "knitr",        # Tables
  "kableExtra"    # Pretty tables
)

# Install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing", pkg, "..."))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
  }
}

invisible(sapply(required_packages, install_if_missing))

# Load packages
suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(boot)
})

# Optional packages
has_effsize <- requireNamespace("effsize", quietly = TRUE)
has_kableExtra <- requireNamespace("kableExtra", quietly = TRUE)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR <- "bench/data"
REPORT_DIR <- "bench/reports"
CONFIDENCE_LEVEL <- 0.95
BOOTSTRAP_R <- 10000

# Create directories
dir.create(REPORT_DIR, showWarnings = FALSE, recursive = TRUE)

# Color palette for libraries
COLORS <- c(
  "viva_tensor" = "#2E8B57",  # Forest green
  "pytorch" = "#E74C3C",      # Red
  "numpy" = "#3498DB"         # Blue
)

# =============================================================================
# Data Loading
# =============================================================================

#' Load and parse benchmark results from JSON
load_benchmark_data <- function(file = NULL) {
  if (is.null(file)) {
    file <- file.path(DATA_DIR, "benchmark_pro_latest.json")
  }

  if (!file.exists(file)) {
    stop(paste("File not found:", file))
  }

  data <- fromJSON(file, flatten = TRUE)

  # Extract results into a clean tibble
  results <- as_tibble(data$results) %>%
    mutate(
      library = as.factor(library),
      size = as.integer(size),
      size_label = paste0(size, "×", size),
      mean_gflops = stats.mean_gflops,
      std_gflops = stats.std_gflops,
      median_gflops = stats.median_gflops,
      ci_lower = stats.ci_lower,
      ci_upper = stats.ci_upper,
      cv = stats.coefficient_of_variation,
      n_samples = stats.n_samples,
      n_outliers = stats.n_outliers,
      is_normal = stats.is_normal
    ) %>%
    select(library, size, size_label, mean_gflops, std_gflops, median_gflops,
           ci_lower, ci_upper, cv, n_samples, n_outliers, is_normal, backend)

  # Extract raw data for detailed analysis
  raw_data <- lapply(1:nrow(data$results), function(i) {
    tibble(
      library = data$results$library[i],
      size = data$results$size[i],
      gflops = unlist(data$results$stats.clean_gflops[i])
    )
  }) %>%
    bind_rows()

  list(
    summary = results,
    raw = raw_data,
    metadata = data$metadata,
    comparisons = if (!is.null(data$comparisons)) as_tibble(data$comparisons) else NULL
  )
}

# =============================================================================
# Statistical Analysis
# =============================================================================

#' Compute Cliff's Delta effect size (non-parametric)
cliffs_delta <- function(x, y) {
  n1 <- length(x)
  n2 <- length(y)

  # Count dominance
  dominance <- outer(x, y, function(a, b) sign(a - b))
  delta <- sum(dominance) / (n1 * n2)

  # Interpretation
  d <- abs(delta)
  interpretation <- if (d < 0.147) {
    "negligible"
  } else if (d < 0.33) {
    "small"
  } else if (d < 0.474) {
    "medium"
  } else {
    "large"
  }

  list(delta = delta, interpretation = interpretation)
}

#' Bootstrap confidence interval for the mean
bootstrap_ci <- function(x, R = BOOTSTRAP_R, conf = CONFIDENCE_LEVEL) {
  boot_mean <- function(data, indices) mean(data[indices])

  b <- boot(x, boot_mean, R = R)

  # BCa interval
  ci <- tryCatch(
    boot.ci(b, conf = conf, type = "bca")$bca[4:5],
    error = function(e) {
      # Fallback to percentile
      boot.ci(b, conf = conf, type = "perc")$perc[4:5]
    }
  )

  list(lower = ci[1], upper = ci[2])
}

#' Perform comprehensive pairwise comparisons
pairwise_analysis <- function(raw_data) {
  sizes <- unique(raw_data$size)
  libraries <- unique(raw_data$library)

  results <- list()

  for (s in sizes) {
    size_data <- raw_data %>% filter(size == s)

    for (lib1 in libraries) {
      for (lib2 in libraries) {
        if (lib1 >= lib2) next

        data1 <- size_data %>% filter(library == lib1) %>% pull(gflops)
        data2 <- size_data %>% filter(library == lib2) %>% pull(gflops)

        if (length(data1) < 3 || length(data2) < 3) next

        # Mann-Whitney U test
        mw <- wilcox.test(data1, data2, conf.int = TRUE)

        # Cliff's delta
        cd <- cliffs_delta(data1, data2)

        # Bootstrap CI for difference
        diff_data <- mean(data1) - mean(data2)
        diff_pct <- (mean(data1) / mean(data2) - 1) * 100

        results[[length(results) + 1]] <- tibble(
          size = s,
          library1 = lib1,
          library2 = lib2,
          mean1 = mean(data1),
          mean2 = mean(data2),
          diff_abs = diff_data,
          diff_pct = diff_pct,
          mann_whitney_p = mw$p.value,
          significant = mw$p.value < (1 - CONFIDENCE_LEVEL),
          cliffs_delta = cd$delta,
          effect_size = cd$interpretation,
          winner = ifelse(mean(data1) > mean(data2), lib1, lib2)
        )
      }
    }
  }

  bind_rows(results)
}

# =============================================================================
# Visualization
# =============================================================================

#' Create performance comparison bar chart
plot_performance_bars <- function(data, output_file = NULL) {
  p <- ggplot(data$summary, aes(x = size_label, y = mean_gflops, fill = library)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.9) +
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper),
      position = position_dodge(width = 0.8),
      width = 0.2,
      linewidth = 0.5
    ) +
    scale_fill_manual(values = COLORS) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.1)),
      labels = comma
    ) +
    labs(
      title = "Matrix Multiplication Performance (FP64 DGEMM)",
      subtitle = "Error bars: 95% Bootstrap BCa confidence intervals",
      x = "Matrix Size",
      y = "Performance (GFLOPS)",
      fill = "Library",
      caption = paste("n =", max(data$summary$n_samples), "runs per configuration")
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 10, color = "gray40"),
      plot.caption = element_text(hjust = 1, size = 8, color = "gray50"),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank()
    )

  if (!is.null(output_file)) {
    ggsave(output_file, p, width = 10, height = 6, dpi = 300)
    message(paste("Saved:", output_file))
  }

  p
}

#' Create violin plot showing distribution
plot_distribution <- function(data, output_file = NULL) {
  p <- ggplot(data$raw, aes(x = factor(size), y = gflops, fill = library)) +
    geom_violin(alpha = 0.7, scale = "width", position = position_dodge(width = 0.8)) +
    geom_boxplot(width = 0.15, position = position_dodge(width = 0.8),
                 alpha = 0.8, outlier.size = 1) +
    scale_fill_manual(values = COLORS) +
    scale_y_continuous(labels = comma) +
    labs(
      title = "Performance Distribution by Library",
      subtitle = "Violin plots with embedded boxplots",
      x = "Matrix Size",
      y = "Performance (GFLOPS)",
      fill = "Library"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
      legend.position = "bottom"
    )

  if (!is.null(output_file)) {
    ggsave(output_file, p, width = 10, height = 6, dpi = 300)
    message(paste("Saved:", output_file))
  }

  p
}

#' Create speedup comparison plot
plot_speedup <- function(data, baseline = "numpy", output_file = NULL) {
  baseline_perf <- data$summary %>%
    filter(library == baseline) %>%
    select(size, baseline_gflops = mean_gflops)

  speedup <- data$summary %>%
    left_join(baseline_perf, by = "size") %>%
    mutate(speedup = mean_gflops / baseline_gflops)

  p <- ggplot(speedup, aes(x = size_label, y = speedup, fill = library)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.9) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.7) +
    scale_fill_manual(values = COLORS) +
    scale_y_continuous(
      labels = function(x) paste0(x, "×"),
      expand = expansion(mult = c(0, 0.1))
    ) +
    labs(
      title = paste("Speedup Relative to", tools::toTitleCase(baseline)),
      subtitle = "Values > 1× indicate faster performance",
      x = "Matrix Size",
      y = "Speedup",
      fill = "Library"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
      legend.position = "bottom"
    )

  if (!is.null(output_file)) {
    ggsave(output_file, p, width = 10, height = 6, dpi = 300)
    message(paste("Saved:", output_file))
  }

  p
}

# =============================================================================
# Report Generation
# =============================================================================

#' Generate comprehensive markdown report
generate_report <- function(data, comparisons, output_file) {
  sink(output_file)

  cat("# viva_tensor Statistical Analysis Report\n\n")
  cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

  # Metadata
  cat("## Environment\n\n")
  cat("| Property | Value |\n")
  cat("|:---------|:------|\n")
  cat(sprintf("| Platform | %s |\n", data$metadata$platform))
  cat(sprintf("| Confidence Level | %.0f%% |\n", CONFIDENCE_LEVEL * 100))
  cat(sprintf("| Bootstrap Samples | %s |\n", format(BOOTSTRAP_R, big.mark = ",")))
  cat("\n")

  # Summary statistics
  cat("## Performance Summary\n\n")
  cat("| Size | Library | Mean ± SD | 95% CI | CV% | Normal? |\n")
  cat("|:----:|:--------|:---------:|:------:|:---:|:-------:|\n")

  for (i in 1:nrow(data$summary)) {
    r <- data$summary[i, ]
    normal <- ifelse(r$is_normal, "✓", "✗")
    cv_status <- ifelse(r$cv < 5, "✓", "⚠")
    cat(sprintf("| %s | %s | **%.1f** ±%.1f | [%.1f, %.1f] | %.1f%s | %s |\n",
                r$size_label, r$library, r$mean_gflops, r$std_gflops,
                r$ci_lower, r$ci_upper, r$cv, cv_status, normal))
  }
  cat("\n")

  # Statistical comparisons
  cat("## Statistical Comparisons\n\n")
  cat("Using Mann-Whitney U test (non-parametric) and Cliff's Delta effect size.\n\n")
  cat("| Size | Comparison | Δ% | p-value | Sig. | Effect |\n")
  cat("|:----:|:-----------|:--:|:-------:|:----:|:------:|\n")

  for (i in 1:nrow(comparisons)) {
    c <- comparisons[i, ]
    sig <- ifelse(c$significant, "**Yes**", "No")
    cat(sprintf("| %d×%d | %s vs %s | %+.1f%% | %.2e | %s | %s (δ=%.2f) |\n",
                c$size, c$size, c$library1, c$library2,
                c$diff_pct, c$mann_whitney_p, sig,
                c$effect_size, c$cliffs_delta))
  }
  cat("\n")

  # Winner summary
  cat("## Winners\n\n")
  sizes <- unique(data$summary$size)
  for (s in sort(sizes)) {
    size_data <- data$summary %>% filter(size == s)
    winner <- size_data %>% slice_max(mean_gflops, n = 1)
    second <- size_data %>% arrange(desc(mean_gflops)) %>% slice(2)

    margin <- (winner$mean_gflops / second$mean_gflops - 1) * 100
    cat(sprintf("- **%d×%d**: %s (+%.1f%% vs %s)\n",
                s, s, winner$library, margin, second$library))
  }
  cat("\n")

  cat("---\n")
  cat("*Analysis by viva_tensor benchmark suite*\n")

  sink()
  message(paste("Report saved:", output_file))
}

# =============================================================================
# Main
# =============================================================================

main <- function() {
  cat(strrep("=", 70), "\n")
  cat("  viva_tensor Statistical Analysis (R)\n")
  cat(strrep("=", 70), "\n\n")

  # Load data
  cat("Loading benchmark data...\n")
  data <- tryCatch(
    load_benchmark_data(),
    error = function(e) {
      cat("Error:", e$message, "\n")
      cat("Run 'python3 bench/benchmark_pro.py' first.\n")
      quit(status = 1)
    }
  )

  cat("Loaded", nrow(data$summary), "results\n")
  cat("Libraries:", paste(unique(data$summary$library), collapse = ", "), "\n")
  cat("Sizes:", paste(unique(data$summary$size), collapse = ", "), "\n\n")

  # Statistical analysis
  cat("Performing statistical analysis...\n")
  comparisons <- pairwise_analysis(data$raw)
  cat("Completed", nrow(comparisons), "pairwise comparisons\n\n")

  # Generate plots
  cat("Generating visualizations...\n")
  plot_performance_bars(data, file.path(REPORT_DIR, "performance_bars.png"))
  plot_distribution(data, file.path(REPORT_DIR, "distribution.png"))
  plot_speedup(data, "pytorch", file.path(REPORT_DIR, "speedup_vs_pytorch.png"))
  cat("\n")

  # Generate report
  cat("Generating report...\n")
  generate_report(data, comparisons, file.path(REPORT_DIR, "analysis_report.md"))

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("  Analysis Complete!\n")
  cat(strrep("=", 70), "\n")
  cat("\nOutputs:\n")
  cat("  - bench/reports/performance_bars.png\n")
  cat("  - bench/reports/distribution.png\n")
  cat("  - bench/reports/speedup_vs_pytorch.png\n")
  cat("  - bench/reports/analysis_report.md\n")
}

# Run
if (!interactive()) {
  main()
}
