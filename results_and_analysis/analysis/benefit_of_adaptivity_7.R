library(ggplot2)
library(tidyverse)
library(RColorBrewer)


# Load data
results_dir <- "../results/"
plots_dir <- "../../plots/v7/"

setwd(results_dir)
studies <- c(
    "7.x_benefits_of_adaptivity/mu-agg_on_knn/mu-agg_on_knn-individual_acc_df.csv",
    "7.x_n_ratings_offline_testing/mu-agg-individual_acc_df.csv",
    "7.x_n_ratings_offline_testing/knn-individual_acc_df.csv"
)

study_names <- c(
    "GP-Agg Random",
    "GP-Agg Adaptive",
    "Nearest Neighbor Random"
)

indiv_results <- list()
for (s in seq_along(studies)) {
    name <- study_names[s]
    res <- read.csv(studies[s])
    res$Model <- name
    indiv_results[[name]] <- res
}

avg_acc_by_ratings <- do.call(
    rbind,
    lapply(
        indiv_results,
        function(x) colMeans(select(x, -Model, -user_id))
    )
)
colnames(avg_acc_by_ratings) <- gsub("X", "", colnames(avg_acc_by_ratings))

# Convert avg_acc_by_ratings to long format
avg_acc_by_ratings_long <- as.data.frame(avg_acc_by_ratings) %>%
    rownames_to_column(var = "Model") %>%
    pivot_longer(cols = -Model, names_to = "Ratings", values_to = "Accuracy")

palette <- c("#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600")
acc_by_n_ratings_lines <- ggplot(
    avg_acc_by_ratings_long,
    aes(x = as.numeric(Ratings), y = Accuracy, color = Model, linetype = Model)
) +
    geom_line(alpha = 0.3) +
    geom_smooth(se = FALSE, span = 0.8) +
    scale_color_manual(values = palette[c(1, 3, 5)]) +
    theme_bw() +
    labs(title = "", x = "Ratings", y = "Accuracy")
acc_by_n_ratings_lines

# setwd(plots_dir)
# pdf("~/Dropbox/Apps/Overleaf/Adaptive Preference Measurement - Working Paper/fig/benefits_from_adaptivity.pdf", height = 3, width = 5.5)
acc_by_n_ratings_lines
# dev.off()
