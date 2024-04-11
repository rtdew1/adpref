library(ggplot2)
library(tidyverse)
library(xtable)
library(RColorBrewer)
library(tidyr)

results_dir <- "../results/ar_results/"
plots_dir <- "../../analysis/plots/v5/"

setwd(results_dir)

studies <- c(
  "redone_41_knn1",
  "analysis-5.0-og-ar-dresses10-None",
  "analysis-5.1.2-aggregated-ar-dresses10-None",
  "analysis-5.8-gur_w_old_priors-redo"
)

study_names <- c(
  "Nearest",
  "MU",
  "MU-Agg",
  "KUR"
)

indiv_results <- list()
for (s in seq_along(studies)) {
  name <- study_names[s]
  res <- read.csv(paste(studies[s], "/individuals.csv", sep = ""))
  res <- res[res$passed_redisplay == 1, ]
  res$Model <- name
  indiv_results[[name]] <- res
}

# Combine all the individual results into a single data frame
combined_indiv_res <- do.call(rbind, indiv_results)

setwd(plots_dir)




# Useful functions

compare_densities <- function(ind_results, var, models, xlab, palette = NA) {
  if (any(is.na(palette))) {
    palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
  }

  compare_densities <- ggplot(
    ind_results[ind_results$Model %in% models, ],
    aes(x = !!sym(var), fill = Model, linetype = Model)
  ) +
    geom_density(alpha = 0.5) +
    geom_vline(xintercept = 0.0, linetype = "dotted") +
    scale_fill_manual(values = palette) +
    labs(x = xlab, y = "Density") +
    theme_bw()

  return(compare_densities)
}

summarize_stat <- function(results_list, col_name, stat_name, stat_func, se_func) {
  stat_df <- data.frame()
  for (s in seq_along(results_list)) {
    name <- names(results_list)[s]
    res <- results_list[[s]]
    stat_df <- rbind(
      stat_df,
      data.frame(
        Model = name,
        Stat = stat_name,
        Value = stat_func(res[, col_name]),
        SE = se_func(res[, col_name])
      )
    )
  }
  return(stat_df)
}

bernoulli_se <- function(x) sqrt(mean(x) * (1 - mean(x)) / length(x))
mean_se <- function(x) t.test(x)$stderr





## Plot the distribution of fixed_test_corr as a density for each model -----------

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
compare_corr_densities <- compare_densities(
  ind_results = combined_indiv_res,
  var = "fixed_test_corr",
  models = c("Nearest", "MU-Agg"),
  xlab = "Correlation",
  palette = palette[c(1, 5)]
)
compare_corr_densities


# pdf(file = "compare_corr_densities.pdf", width = 5, height = 3)
compare_corr_densities
# dev.off()


palette <- c("#00429d", "#4771b2", "#73a2c6", "#a5d5d8", "#ffffe0")
compare_acfn_corr_densities <- compare_densities(
  ind_results = combined_indiv_res,
  var = "fixed_test_corr",
  models = c("MU", "MU-Agg", "KUR"),
  xlab = "Correlation",
  palette = palette[c(1, 3, 5)]
)
compare_acfn_corr_densities

# pdf(file = "compare_acfn_corr_densities.pdf", width = 5, height = 3)
compare_acfn_corr_densities
# dev.off()



## Plot the distribution of fixed_test_rmse as a density for each model -----------




## Plot the distribution of fixed_test_corr as a density for each model -----------

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
compare_rmse_densities <- compare_densities(
  ind_results = combined_indiv_res,
  var = "fixed_test_rmse",
  models = c("Nearest", "MU-Agg"),
  xlab = "RMSE",
  palette = palette[c(1, 5)]
)
compare_rmse_densities


# pdf(file = "compare_rmse_densities.pdf", width = 5, height = 3)
compare_rmse_densities
# dev.off()


palette <- c("#00429d", "#4771b2", "#73a2c6", "#a5d5d8", "#ffffe0")
compare_acfn_rmse_densities <- compare_densities(
  ind_results = combined_indiv_res,
  var = "fixed_test_rmse",
  models = c("MU", "MU-Agg", "KUR"),
  xlab = "RMSE",
  palette = palette[c(1, 3, 5)]
)
compare_acfn_rmse_densities

# pdf(file = "compare_acfn_rmse_densities.pdf", width = 5, height = 3)
compare_acfn_rmse_densities
# dev.off()
# 





## Compare individual-level stats by model --------------------------------------

corr_stats_df <- summarize_stat(
  indiv_results,
  "fixed_test_corr",
  "Random Items, Correlation",
  stat_func = function(x) mean(x, na.rm = TRUE),
  se_func = mean_se
)

corr_stats_df$Model <- factor(corr_stats_df$Model, levels = study_names)
avg_corr_plot_errors <- ggplot(corr_stats_df) +
  geom_col(aes(x = Model, y = Value), fill = "#73a2c6") +
  geom_errorbar(
    aes(x = Model, ymax = Value + 2 * SE, ymin = Value - 2 * SE),
    color = "#00429d",
    width = 0.25
  ) +
  geom_hline(yintercept = 0.5, linetype = "dotted") +
  ylab("Average Correlation") +
  xlab("Model") +
  theme_bw()

# pdf(file = "avg_corr_plot_errors.pdf", width = 5, height = 3)
avg_corr_plot_errors
# dev.off()

# Check ANOVA results, just to be sure
summary(aov(fixed_test_corr ~ Model, data = combined_indiv_res))



## Compare GOOD accuracies by model --------------------------------------------

# Choose models
models_to_compare <- c("Nearest", "MU-Agg")

# Compute min/max accuracies
min_acc <- min(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$ratings_accuracy)
max_acc <- max(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$ratings_accuracy)

# Create df for plotting
ratings_acc_plot_df <- data.frame()
for (name in models_to_compare) {
  res <- indiv_results[[name]]
  acc <- table(factor(res$ratings_accuracy, levels = seq(min_acc, max_acc, 0.2))) / nrow(res)
  ratings_acc_plot_df <- rbind(
    ratings_acc_plot_df,
    data.frame(Model = name, Mean = acc)
  )
}
colnames(ratings_acc_plot_df) <- c("Model", "Accuracy", "Proportion")
ratings_acc_plot_df$Model <- factor(ratings_acc_plot_df$Model, levels = c("Nearest", "Adaptive Linear", "MU-Agg"))

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
ratings_acc_by_model_columns <- ggplot(
  ratings_acc_plot_df,
  aes(x = Accuracy, y = Proportion, fill = Model)
) +
  geom_col(position = "dodge") +
  scale_x_discrete(labels = seq(min_acc, max_acc, by = 0.2), drop = FALSE) +
  scale_fill_manual(values = palette[c(1, 5)]) +
  theme_bw()
ratings_acc_by_model_columns

# pdf(file = "ratings_acc_by_model_columns.pdf", width = 6, height = 3)
ratings_acc_by_model_columns
# dev.off()


## Create a table with combined results

# Average corr (same as above)
corr_stats_df <- summarize_stat(
  indiv_results,
  "fixed_test_corr",
  "Random Items, Correlation",
  stat_func = function(x) mean(x, na.rm = TRUE),
  se_func = mean_se
)

# Average RMSE
rmse_stats_df <- summarize_stat(
  indiv_results,
  "fixed_test_rmse",
  "Random Items, RMSE",
  stat_func = function(x) mean(x, na.rm = TRUE),
  se_func = mean_se
)

# Average good accuracy
acc_stats_df <- summarize_stat(
  indiv_results,
  "ratings_accuracy",
  "Select Good Items, Accuracy",
  stat_func = function(x) mean(x, na.rm = TRUE),
  se_func = mean_se
)

# Combine everything
all_stats_df <- rbind(
  corr_stats_df,
  rmse_stats_df,
  acc_stats_df
)
stat_df <- pivot_wider(all_stats_df[, c("Model", "Stat", "Value")],
  names_from = Stat,
  values_from = Value
)
stat_df$What <- "Stat"
se_df <- pivot_wider(all_stats_df[, c("Model", "Stat", "SE")],
  names_from = Stat,
  values_from = SE
)
se_df$What <- "SE"
combined_df <- rbind(stat_df, se_df)
combined_df <- combined_df %>%
  select(Model, What, everything()) %>%
  mutate(Model = factor(Model, levels = c("Nearest", "MU", "MU-Agg", "KUR"))) %>%
  arrange(Model)

# Save using xtable
print(
  xtable(combined_df, digits = 3, caption = "Individual-level results for each model."),
  type = "latex",
  booktabs = TRUE,
  include.rownames = FALSE,
  file = "indiv_results.tex"
)
