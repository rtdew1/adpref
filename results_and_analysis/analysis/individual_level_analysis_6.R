library(ggplot2)
library(tidyverse)
library(xtable)
library(RColorBrewer)
library(tidyr)

results_dir <- "../results/mpc_results/"
plots_dir <- "../../analysis/plots/v6/"

setwd(results_dir)

studies <- c(
  "redone-6.2_knn1_no_renorm",
  "analysis-6.8-mpc_fixed_og_new_priors-None",
  "analysis-6.9-mpc_fixed_gur_sym-None"
)

study_names <- c(
  "Nearest",
  "MU",
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
  models = study_names,
  xlab = "Correlation",
  palette = palette[c(1, 3, 5)]
)
compare_corr_densities


# pdf(file = "compare_corr_densities.pdf", width = 5, height = 3)
compare_corr_densities
# dev.off()




# Choose models
models_to_compare <- study_names

# Compute min/max accuracies
min_acc <- min(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$fixed_mpc_accuracy)
max_acc <- max(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$fixed_mpc_accuracy)

# Create df for plotting
acc_plot_df <- data.frame()
for (name in models_to_compare) {
  res <- indiv_results[[name]]
  acc <- table(factor(res$fixed_mpc_accuracy, levels = seq(min_acc, max_acc, 0.1))) / nrow(res)
  acc_plot_df <- rbind(
    acc_plot_df,
    data.frame(Model = name, Mean = acc)
  )
}
colnames(acc_plot_df) <- c("Model", "Accuracy", "Proportion")
acc_plot_df$Model <- factor(acc_plot_df$Model, levels = models_to_compare)

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
acc_by_model_columns <- ggplot(
  acc_plot_df,
  aes(x = Accuracy, y = Proportion, fill = Model)
) +
  geom_col(position = "dodge") +
  scale_x_discrete(labels = seq(min_acc, max_acc, by = 0.1), drop = FALSE) +
  scale_fill_manual(values = palette[c(1, 3, 5)]) +
  theme_bw()

# pdf(file = "acc_by_model_columns.pdf", width = 7, height = 3)
acc_by_model_columns
# dev.off()


## Plot the distribution of fixed_mpc_accuracy as a density for each model -----------

focal_results <- combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]
focal_results$Model <- factor(focal_results$Model)

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
compare_acc_densities <- ggplot(
  focal_results,
  aes(x = fixed_mpc_accuracy, fill = Model, linetype = Model)
) +
  geom_density(alpha = 0.5, bw = 0.05) +
  scale_fill_manual(values = palette[c(1, 3, 5)]) +
  geom_vline(xintercept = 0.5, linetype = "dotted") +
  labs(x = "Accuracy", y = "Density") +
  theme_bw()

# pdf(file = "compare_acc_densities.pdf", width = 5, height = 3)
compare_acc_densities
# dev.off()



acc_stats_df <- data.frame()
for (s in seq_along(indiv_results)) {
  name <- names(indiv_results)[s]
  res <- indiv_results[[s]]
  acc_stats_df <- rbind(
    acc_stats_df,
    data.frame(
      Model = name,
      Stat = "Random Pairs, Accuracy",
      Value = mean(res$fixed_mpc_accuracy),
      SE = t.test(res$fixed_mpc_accuracy)$stderr
    )
  )
}

acc_stats_df$Model <- factor(acc_stats_df$Model, levels = study_names)
avg_acc_plot_errors <- ggplot(acc_stats_df) +
  geom_col(aes(x = Model, y = Value),
    fill = "#73a2c6"
  ) +
  geom_errorbar(aes(x = Model, ymax = Value + 2 * SE, ymin = Value - 2 * SE),
    color = "#00429d",
    width = 0.25
  ) +
  geom_hline(yintercept = 0.5, linetype = "dotted") +
  ylab("Average Accuracy") +
  xlab("Model") +
  theme_bw()

# pdf(file = "avg_acc_plot_errors.pdf", width = 5, height = 3)
avg_acc_plot_errors
# dev.off()

# Check ANOVA results, just to be sure
summary(aov(fixed_mpc_accuracy ~ Model, data = combined_indiv_res))


# Average accuracy (same as above)
acc_stats_df <- summarize_stat(
  indiv_results,
  "fixed_mpc_accuracy",
  "Random Pairs, Accuracy",
  stat_func = mean,
  se_func = mean_se
)

# Percent above 0.5
p_above_half_df <- summarize_stat(
  indiv_results,
  "fixed_mpc_accuracy",
  "Random Pairs, Above 0.5",
  stat_func = function(x) mean(x > 0.5),
  se_func = function(x) bernoulli_se(x > 0.5)
)

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

all_stats_df <- rbind(
  acc_stats_df,
  p_above_half_df,
  corr_stats_df,
  rmse_stats_df
)



# Pivot the table to get separate columns for Value and SE
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

# Reorder the columns
combined_df <- combined_df %>%
  select(Model, What, everything()) %>%
  mutate(Model = factor(Model, levels = study_names)) %>%
  arrange(Model)

print(
  xtable(combined_df, digits = 3, caption = "Individual-level results for each model."),
  type = "latex",
  booktabs = TRUE,
  include.rownames = FALSE,
)
