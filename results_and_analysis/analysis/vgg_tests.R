# ChatGPT-4 assisted
# Required Libraries
# Make sure to change wd to current file
library(ggplot2)
library(tidyr)

mu_agg_vgg_offline <- read.csv("../results/7.x_different_reps/mu-agg_vgg_offline.csv")
vgg_online <- read.csv("../results/7.x_n_ratings_offline_testing/vgg-individual_acc_df.csv", row.names = 1)
mu_agg_online <- read.csv("../results/7.x_n_ratings_offline_testing/mu-agg-individual_acc_df.csv", row.names = 1)

data <- cbind(mu_agg_vgg_offline, colMeans(vgg_online))
data$Ratings <- 5:41
colnames(data) <- c("RecSys", "VGG-PCA-10 (Offline)", "VGG-PCA-10 (Online)", "Ratings")

# Convert data to long format for plotting
data_long <- pivot_longer(data, -Ratings, names_to = "Data", values_to = "Accuracy")

# Make line plot
palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
diff_reps_adaptive_items_plot <- ggplot(
  data_long,
  aes(x = Ratings, y = Accuracy, color = Data, group = Data, linetype = Data)
) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE, span = 0.8) +
  scale_color_manual(values = palette[c(1, 3, 5)]) +
  labs(
    title = "",
    x = "Number of Ratings",
    y = "Accuracy",
    color = "Data"
  ) +
  ylim(0.5, NA) +
  theme_bw()

# Save plot
# setwd("../code/analysis/plots/v7/")
# pdf("vgg_adaptive_learning.pdf", height = 4, width = 7)
# diff_reps_adaptive_items_plot
# dev.off()



# Density plots

results_dir <- "../results/binary_mpc_results/"
plots_dir <- "plots/v7"

setwd(results_dir)

studies <- c(
  "analysis-7.3-aggregated-fixed-binary-mpc-None",
  "analysis-7.9-agg-binmpc-vgg-None"
)

study_names <- c(
  "RecSys",
  "VGG-PCA-10"
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
models_to_compare <- study_names

# Compute min/max accuracies
min_acc <- min(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$fixed_mpc_accuracy)
max_acc <- max(combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]$fixed_mpc_accuracy)

focal_results <- combined_indiv_res[combined_indiv_res$Model %in% models_to_compare, ]
focal_results$Model <- factor(focal_results$Model, levels = study_names)

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
compare_acc_densities <- ggplot(
  focal_results,
  aes(x = fixed_mpc_accuracy, fill = Model, linetype = Model)
) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = palette[c(1, 5)]) +
  geom_vline(xintercept = 0.5, linetype = "dotted") +
  labs(x = "Accuracy", y = "Density") +
  theme_bw()

# pdf(file = "vgg_acc_densities.pdf", width = 5, height = 3)
# compare_acc_densities
# dev.off()
