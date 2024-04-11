library(ggplot2)
library(tidyverse)
library(xtable)
library(RColorBrewer)
library(tidyr)

results_dir <- "../results/binary_mpc_results/"
plots_dir <- "../../analysis/plots/v7/"

setwd(results_dir)

studies <- c(
  "analysis-7.0-knn1-binary-mpc-None",
  "analysis-7.1.2-abernethy-binary-mpc-None",
  "analysis-7.3-aggregated-fixed-binary-mpc-None",
  "analysis-7.4-og-binary-mpc-None",
  "analysis-7.8-gur_approx_sym-binary-mpc-update1-new_cand-None"
)

study_names <- c(
  "NN",
  "Adaptive Linear",
  "MU-Agg",
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


## Compare individual-level accuracies by model ---------------------------------

# Choose models
models_to_compare <- c("NN", "Adaptive Linear", "MU-Agg")

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
acc_plot_df$Model <- factor(acc_plot_df$Model, levels = c("NN", "Adaptive Linear", "MU-Agg"))

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
focal_results$Model <- factor(focal_results$Model, levels = c("NN", "Adaptive Linear", "MU-Agg"))

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
compare_acc_densities <- ggplot(
  focal_results,
  aes(x = fixed_mpc_accuracy, fill = Model, linetype = Model)
) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = palette[c(1, 3, 5)]) +
  geom_vline(xintercept = 0.5, linetype = "dotted") +
  labs(x = "Accuracy", y = "Density") +
  theme_bw()

# pdf(file = "compare_acc_densities.pdf", width = 5, height = 3)
compare_acc_densities
# dev.off()

palette <- c("#00429d", "#4771b2", "#73a2c6", "#a5d5d8", "#ffffe0")
compare_acc_dens_mu_KUR <- ggplot(
  combined_indiv_res[
    combined_indiv_res$Model %in% c("MU", "MU-Agg", "KUR"),
  ],
  aes(x = fixed_mpc_accuracy, fill = Model, linetype = Model)
) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = palette[c(1, 3, 5)]) +
  geom_vline(xintercept = 0.5, linetype = "dotted") +
  labs(x = "Accuracy", y = "Density") +
  theme_bw()

# pdf(file = "compare_acc_densities_mu_KUR.pdf", width = 5, height = 3)
compare_acc_dens_mu_KUR
# dev.off()


## Compare individual-level stats by model --------------------------------------

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

acc_stats_df$Model <- factor(acc_stats_df$Model, levels = c("NN", "Adaptive Linear", "MU", "MU-Agg", "KUR"))
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



## Compare GOOD accuracies by model --------------------------------------------

# Choose models
models_to_compare <- c("NN", "Adaptive Linear", "MU-Agg")

# Compute min/max accuracies
min_acc <- min(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$good_mpc_accuracy)
max_acc <- max(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$good_mpc_accuracy)

# Create df for plotting
good_acc_plot_df <- data.frame()
for (name in models_to_compare) {
  res <- indiv_results[[name]]
  acc <- table(factor(res$good_mpc_accuracy, levels = seq(min_acc, max_acc, 0.2))) / nrow(res)
  good_acc_plot_df <- rbind(
    good_acc_plot_df,
    data.frame(Model = name, Mean = acc)
  )
}
colnames(good_acc_plot_df) <- c("Model", "Accuracy", "Proportion")
good_acc_plot_df$Model <- factor(good_acc_plot_df$Model, levels = c("NN", "Adaptive Linear", "MU-Agg"))

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
good_acc_by_model_columns <- ggplot(
  good_acc_plot_df,
  aes(x = Accuracy, y = Proportion, fill = Model)
) +
  geom_col(position = "dodge") +
  scale_x_discrete(labels = seq(min_acc, max_acc, by = 0.2), drop = FALSE) +
  scale_fill_manual(values = palette[c(1, 3, 5)]) +
  theme_bw()

# pdf(file = "good_acc_by_model_columns.pdf", width = 6, height = 3)
good_acc_by_model_columns
# dev.off()


## Create a table with combined results

# Useful functions
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

# Average good accuracy
good_acc_stats_df <- summarize_stat(
  indiv_results,
  "good_mpc_accuracy",
  "Select Pairs, Accuracy",
  stat_func = mean,
  se_func = mean_se
)

# Percent above 0.5 for good
good_p_above_half_df <- summarize_stat(
  indiv_results,
  "good_mpc_accuracy",
  "Select Pairs, Above 0.5",
  stat_func = function(x) mean(x > 0.5),
  se_func = function(x) bernoulli_se(x > 0.5)
)

all_stats_df <- rbind(
  acc_stats_df,
  p_above_half_df,
  good_acc_stats_df,
  good_p_above_half_df
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
  mutate(Model = factor(Model, levels = c("NN", "Adaptive Linear", "MU", "MU-Agg", "KUR"))) %>%
  arrange(Model)

print(
  xtable(combined_df, digits = 3, caption = "Individual-level results for each model."),
  type = "latex",
  booktabs = TRUE,
  include.rownames = FALSE,
)
