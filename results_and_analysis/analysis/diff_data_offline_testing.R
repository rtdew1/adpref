# ChatGPT-4 assisted
# Required Libraries
library(ggplot2)
library(tidyr)

## MU-AGG, ADAPTIVE ----------------------------------------------
setwd("../results/7.x_different_reps/")
data <- read.csv("mu-agg_diff_reps_adaptive_items.csv")
data$Ratings <- 5:41
data <- data[, c("Original", "PCA.5", "VGG.PCA.10", "Ratings")]

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
    title = "Accuracy as a function of the number of ratings",
    x = "Number of Ratings",
    y = "Accuracy",
    color = "Data"
  ) +
  ylim(0.5, NA) +
  theme_bw()

# Save plot
# setwd("~/Dropbox/1_proj/urbn/code/analysis/plots/v7/")
# pdf("mu_agg_diff_reps_adaptive_items.pdf", height = 4, width = 7)
diff_reps_adaptive_items_plot
# dev.off()



## MU-AGG, RANDOM --------------------------------------------
data <- read.csv("mu-agg_diff_reps_random_items.csv")
data$Ratings <- 5:41
data <- data[, c("Original", "PCA.5", "VGG.PCA.10", "Ratings")]

# Convert data to long format for plotting
data_long <- pivot_longer(data, -Ratings, names_to = "Data", values_to = "Accuracy")

# Make line plot
diff_reps_random_items_plot <- ggplot(data_long, aes(x = Ratings, y = Accuracy, color = Data, group = Data, linetype = Data)) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE, span = 0.8) +
  labs(
    title = "Accuracy as a function of the number of ratings",
    x = "Number of Ratings",
    y = "Accuracy",
    color = "Data"
  ) +
  ylim(0.5, NA) +
  theme_bw()

# Save plot
# setwd("~/Dropbox/1_proj/urbn/code/analysis/plots/v7/")
# pdf("mu_agg_diff_reps_random_items.pdf", height = 4, width = 7)
diff_reps_random_items_plot
# dev.off()



# MU, ADAPTIVE ----------------------------------------------
data <- read.csv("mu_diff_reps_adaptive_items.csv")
data$Ratings <- 5:41

# Convert data to long format for plotting
data_long <- pivot_longer(data, -Ratings, names_to = "Data", values_to = "Accuracy")

# Make line plot
diff_reps_adaptive_items_plot <- ggplot(data_long, aes(x = Ratings, y = Accuracy, color = Data, group = Data, linetype = Data)) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE, span = 0.8) +
  labs(
    title = "Adaptive Items (MU), Individual-level Hyperparameters",
    x = "Number of Ratings",
    y = "Accuracy",
    color = "Data"
  ) +
  ylim(0.5, NA) +
  theme_bw()

# Save plot
# setwd("~/Dropbox/1_proj/urbn/code/analysis/plots/v7/")
# pdf("mu_diff_reps_adaptive_items.pdf", height = 4, width = 7)
diff_reps_adaptive_items_plot
# dev.off()

adaptive_data = data_long
adaptive_data$Selection = "Adaptive (MU)"

# MU, RANDOM --------------------------------------------
data <- read.csv("mu_diff_reps_random_items.csv")
data$Ratings <- 5:41

# Convert data to long format for plotting
data_long <- pivot_longer(data, -Ratings, names_to = "Data", values_to = "Accuracy")

# Make line plot
diff_reps_random_items_plot <- ggplot(data_long, aes(x = Ratings, y = Accuracy, color = Data, group = Data, linetype = Data)) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE, span = 0.8) +
  labs(
    title = "Individual-level Hyperparameters, Random Items",
    x = "Number of Ratings",
    y = "Accuracy",
    color = "Data"
  ) +
  ylim(0.5, NA) +
  theme_bw()

# Save plot
# setwd("~/Dropbox/1_proj/urbn/code/analysis/plots/v7/")
# pdf("mu_diff_reps_random_items.pdf", height = 4, width = 7)
diff_reps_random_items_plot
# dev.off()

random_data = data_long
random_data$Selection = "Random"
combined = rbind(adaptive_data, random_data)
combined = combined[combined$Selection == "Adaptive (MU)",]
palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
faceted_comparison = ggplot(combined, aes(x = Ratings, y = Accuracy, color = Data, group = Data, linetype = Data)) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE, span = 0.8) +
  labs(
    x = "Number of Ratings",
    y = "Accuracy",
    color = "Data"
  ) +
  scale_color_manual(values = palette) + 
  ylim(0.5, NA) +
  #facet_wrap(~Selection) +
  theme_bw()

# pdf("faceted_comparison_diff_reps.pdf", height = 3, width = 5.5)
faceted_comparison
# dev.off()
