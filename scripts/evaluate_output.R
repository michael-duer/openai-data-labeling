library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(grid)

# This script creates different plots to asses and compare the performance of 
# different models and prompts on different/the same samples.

setwd("~/Documents/GitHub/openai-data-labeling/scripts")

# Load file to create testsets from
data <- read.csv("../data/evaluation/evaluation.csv",sep = ",")

# Compare sample
sample_comparison <- c("random_sample_small_3.csv","random_sample_small_4.csv",
                       "random_sample_small_5.csv","random_sample_small_6.csv",
                       "random_sample_small_7.csv","random_sample_small_8.csv",
                       "random_sample_small_9.csv","random_sample_small_10.csv",
                       "random_sample_small_11.csv","random_sample_small_12.csv")
prompt_comparison <- c("refined_prompt3.txt","refined_prompt3_alt.txt")
sample_small1 <- data %>%
  #filter(dataset %in% sample_comparison) %>%
  filter(dataset=="random_sample_small_3.csv") %>%
  #filter(dataset=="synthetic_sample.csv") %>%
  filter(prompt %in% prompt_comparison) %>%
  filter(model=="o1-mini") %>%
  #filter(model!="gpt-3.5-turbo") %>%
  group_by(model, prompt) %>%
  slice_tail(n = 1)

detailed_comparison <- ggplot(sample_small1, aes(x = krippendorff_detailed, y = accuracy_detailed, color = prompt, shape = model)) +
  geom_point(size = 3) +
  labs(
    title = "Detailed Labels",
    x = "Krippendorff's Alpha",
    y = "Accuracy",
    color = "Prompt",
    shape = "Model"
  ) +
  theme(legend.position = "none") +
  theme_minimal()

simplified_comparison <- ggplot(sample_small1, aes(x = krippendorff_simplified, y = accuracy_simplified, color = prompt, shape = model)) +
  geom_point(size = 3) +
  labs(
    title = "Simplified Labels",
    x = "Krippendorff's Alpha",
    y = "Accuracy",
    color = "Prompt",
    shape = "Model"
  ) +
  theme_minimal()

grid.arrange(
  detailed_comparison, simplified_comparison,
  ncol = 2,
  top = textGrob("Accuracy vs Krippendorff's Alpha", gp = gpar(fontsize = 16, fontface = "bold"))
)

## Statistics comparison on 10 sample sets
sample_small_summary <- data %>%
  mutate(row_id = row_number()) %>% # Add ID to later keep the latest observation
  filter(dataset %in% sample_comparison) %>%
  filter(model %in% c("o1-mini","o3-mini","gpt-4.5-preview-2025-02-27","gpt-4-turbo")) %>%
  filter(prompt == "refined_prompt3.txt") %>%
  group_by(model, prompt, dataset) %>%
  slice_max(order_by = row_id, n = 1) %>%
  ungroup() %>%
  select(-dataset, -sample_size, -row_id) %>%
  group_by(model, prompt) %>%
  summarise(across(
    .cols = names(.)[3:ncol(.)],  # skip the first two columns (model, prompt)
    .fns = list(mean = mean, var = var), # calculate mean and variance for all columns
    .names = "{.col}_{.fn}" # name new columns indicating mean/variance
  ))

sample_small_summary_long <- sample_small_summary %>%
  select(model, ends_with("_var")) %>%
  pivot_longer(cols = ends_with("_var"),
               names_to = "metric",
               values_to = "variance")

## TODO: Create Histograms
ggplot(sample_small_summary_long, aes(x = variance)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  labs(title = "Histogram of Metric Variances Across Models",
       x = "Variance",
       y = "Count") +
  theme_minimal()
# Facet by metric
ggplot(sample_small_summary_long, aes(x = variance)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  facet_wrap(~ metric, scales = "free") +
  labs(title = "Variance per Metric", x = "Variance", y = "Count") +
  theme_minimal()
# Facet by model
ggplot(sample_small_summary_long, aes(x = variance)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  facet_wrap(~ model) +
  labs(title = "Variance Distribution per Model",
       x = "Variance",
       y = "Count") +
  theme_minimal()

# Combine data from 10 samples (n=50) and 4 AI Models
model_comparison <- data %>%
  mutate(row_id = row_number()) %>% # Add ID to later keep the latest observation
  filter(dataset %in% sample_comparison,
         model %in% c("gpt-4.1","gpt-4.1-mini","gpt-4.1-nano"),
         #model %in% c("o1-mini","o3-mini","gpt-4.5-preview-2025-02-27","gpt-4-turbo"),
         prompt == "refined_prompt3.txt") %>%
  group_by(model, prompt, dataset) %>%
  slice_max(order_by = row_id, n = 1) %>%
  ungroup()

model_comparison_big <- data %>%
  mutate(row_id = row_number()) %>% # Add ID to later keep the latest observation
  filter(dataset %in% c("random_sample_11.csv", "random_sample_12.csv"),
         model %in% c("gpt-4.1","gpt-4.1-mini","gpt-4.1-nano"),
         #model %in% c("o1-mini","o3-mini","gpt-4.5-preview-2025-02-27","gpt-4-turbo"),
         prompt == "refined_prompt3.txt") %>%
  group_by(model, prompt, dataset) %>%
  slice_max(order_by = row_id, n = 1) %>%
  ungroup()
# Calculate the means separatly
means <- model_comparison %>%
  group_by(model) %>%
  summarise(mean_acc_det = mean(accuracy_detailed),
            mean_acc_sim = mean(accuracy_simplified),
            mean_kripp_det = mean(krippendorff_detailed),
            mean_kripp_sim = mean(krippendorff_simplified))

means_500 <- data %>%
  filter(dataset %in% c("random_sample_11.csv", "random_sample_12.csv"),
         model %in% c("gpt-4.1","gpt-4.1-mini","gpt-4.1-nano"),
         #model %in% c("o1-mini", "o3-mini", "gpt-4.5-preview-2025-02-27", "gpt-4-turbo"),
         prompt == "refined_prompt3.txt") %>%
  group_by(model, dataset) %>%
  slice_max(order_by = row_number(), n = 1) %>%
  group_by(model) %>%
  summarise(mean500_acc_det = mean(accuracy_detailed),
            mean500_acc_sim = mean(accuracy_simplified),
            mean500_kripp_det = mean(krippendorff_detailed),
            mean500_kripp_sim = mean(krippendorff_simplified))



# Histogram Accuracy detailed 
ggplot(model_comparison, aes(x = accuracy_detailed)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  geom_histogram(data = model_comparison_big,aes(x = accuracy_detailed),
                 binwidth = 1,fill = "orange",color = "white") +
  facet_wrap(~ model, ncol = 1) +
  geom_vline(data = means, aes(xintercept = mean_acc_det), 
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(data = means_500, aes(xintercept = mean500_acc_det),
             color = "limegreen", linetype = "dashed", linewidth = 1) +
  coord_cartesian(xlim = c(0, 100)) +
  labs(title = "Accuracy Distribution across 10 Samples (detailed labels)",
       x = "Accuracy in %",
       y = "Number of samples") +
  theme_minimal()
# Histogram Accuracy simplified 
ggplot(model_comparison, aes(x = accuracy_simplified)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  geom_histogram(data = model_comparison_big,aes(x = accuracy_simplified),
                 binwidth = 1,fill = "orange",color = "white") +
  facet_wrap(~ model, ncol = 1) +
  geom_vline(data = means, aes(xintercept = mean_acc_sim), 
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(data = means_500, aes(xintercept = mean500_acc_sim),
             color = "limegreen", linetype = "dashed", linewidth = 1) +
  coord_cartesian(xlim = c(0, 100)) +
  labs(title = "Accuracy Distribution across 10 Samples (simplified labels)",
       x = "Accuracy in %",
       y = "Number of samples") +
  theme_minimal()
# Histogram Accuracy detailed 
ggplot(model_comparison, aes(x = krippendorff_detailed)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
  geom_histogram(data = model_comparison_big,aes(x = krippendorff_detailed),
                 binwidth = 0.05,fill = "orange",color = "white") +
  facet_wrap(~ model, ncol = 1) +
  geom_vline(data = means, aes(xintercept = mean_kripp_det), 
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(data = means_500, aes(xintercept = mean500_kripp_det),
             color = "limegreen", linetype = "dashed", linewidth = 1) +
  coord_cartesian(xlim = c(0, 1)) +
  labs(title = "Krippendorff's Alpha Distribution across 10 Samples (detailed labels)",
       x = "Accuracy in %",
       y = "Number of samples") +
  theme_minimal()
# Histogram Accuracy detailed 
ggplot(model_comparison, aes(x = krippendorff_simplified)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
  geom_histogram(data = model_comparison_big,aes(x = krippendorff_simplified),
                 binwidth = 0.05,fill = "orange",color = "white") +
  facet_wrap(~ model, ncol = 1) +
  geom_vline(data = means, aes(xintercept = mean_kripp_sim), 
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(data = means_500, aes(xintercept = mean500_kripp_sim),
             color = "limegreen", linetype = "dashed", linewidth = 1) +
  coord_cartesian(xlim = c(0, 1)) +
  labs(title = "Krippendorff's Alpha Distribution across 10 Samples (simplified labels)",
       x = "Accuracy in %",
       y = "Number of samples") +
  theme_minimal()

# Create a plot consisting of a subplot for accuracy, k alpha and bp alpha
# Axis: x count, y element
# Color indicating prompt
# Shape indicating model

# Scatter plot containing 2 subplots showing different alphas
# shape: model
# color prompt
# axis: x:accuracy, y:k/bp alpha

# Create first histogram
detailed_accuracy <- ggplot(data, aes(x = accuracy_detailed)) +
  geom_histogram(fill = "steelblue", color = "black", bins = 30) +
  labs(title = "Detailed Labels", x = "Accuracy", y = "Count") +
  theme_minimal()

# Create second histogram
simplified_accuracy <- ggplot(data, aes(x = accuracy_simplified)) +
  geom_histogram(fill = "tomato", color = "black", bins = 30) +
  labs(title = "Simplified Labels", x = "Accuracy", y = "Count") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  detailed_accuracy, simplified_accuracy,
  ncol = 2,
  top = textGrob("Accuracy distribution", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)

#--------------------#
#### Create Plots ####
#--------------------#

#----------------------------#
##### Histogram Accuracy #####
#----------------------------#

# Create first histogram
detailed_accuracy <- ggplot(data, aes(x = accuracy_detailed)) +
  geom_histogram(fill = "steelblue", color = "black", bins = 30) +
  labs(title = "Detailed Labels", x = "Accuracy", y = "Count") +
  theme_minimal()

# Create second histogram
simplified_accuracy <- ggplot(data, aes(x = accuracy_simplified)) +
  geom_histogram(fill = "tomato", color = "black", bins = 30) +
  labs(title = "Simplified Labels", x = "Accuracy", y = "Count") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  detailed_accuracy, simplified_accuracy,
  ncol = 2,
  top = textGrob("Accuracy distribution", gp = gpar(fontsize = 16, fontface = "bold"))#,
  #bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)

#---------------------------------------#
##### Histogram Krippendorff's Alpha #####
#---------------------------------------#

# Create first histogram
detailed_krippendorff <- ggplot(data, aes(x = krippendorff_detailed)) +
  geom_histogram(fill = "steelblue", color = "black", bins = 30) +
  labs(title = "Detailed Labels", x = "Krippendorff's Alpha", y = "Count") +
  theme_minimal()

# Create second histogram
simplified_krippendorff <- ggplot(data, aes(x = krippendorff_simplified)) +
  geom_histogram(fill = "tomato", color = "black", bins = 30) +
  labs(title = "Simplified Labels", x = "Krippendorff's Alpha", y = "Count") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  detailed_krippendorff, simplified_krippendorff,
  ncol = 2,
  top = textGrob("Krippendorff's Alpha distribution", gp = gpar(fontsize = 16, fontface = "bold"))#,
  #bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)

#------------------------------------------#
##### Histogram Brennan-Prediger Alpha #####
#------------------------------------------#

# Create first histogram
detailed_bp <- ggplot(data, aes(x = bp_detailed)) +
  geom_histogram(fill = "steelblue", color = "black", bins = 30) +
  labs(title = "Detailed Labels", x = "Brennan-Prediger Alpha", y = "Count") +
  theme_minimal()

# Create second histogram
simplified_bp <- ggplot(data, aes(x = bp_simplified)) +
  geom_histogram(fill = "tomato", color = "black", bins = 30) +
  labs(title = "Simplified Labels", x = "Brennan-Prediger Alpha", y = "Count") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  detailed_bp, simplified_bp,
  ncol = 2,
  top = textGrob("Brennan-Prediger Alpha distribution", gp = gpar(fontsize = 16, fontface = "bold"))#,
  #bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)

#----------------------------------------------------------------------#
##### Scatterplot: Krippendorff's Alpha and Brennan-Prediger Alpha #####
#----------------------------------------------------------------------#

# Create first histogram
scatterplot_detailed_kripp_bp <- ggplot(data, aes(x = krippendorff_detailed, y = bp_detailed)) +
  geom_point(size = 3) +
  labs(title = "Detailed Labels", x = "Krippendorff's Alpha", y = "Brennan-Prediger Alpha") +
  theme_minimal()

# Create second histogram
scatterplot_simplified_kripp_bp <- ggplot(data, aes(x = krippendorff_simplified, y = bp_simplified)) +
  geom_point(size = 3) +
  labs(title = "Simplified Labels", x = "Krippendorff's Alpha", y = "Brennan-Prediger Alpha") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  scatterplot_detailed_kripp_bp, scatterplot_simplified_kripp_bp,
  ncol = 2,
  top = textGrob("Krippendorff's Alpha vs Brennan-Prediger Alpha", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)

#--------------------------------------------------------#
##### Scatterplot: Krippendorff's Alpha and Accuracy #####
#--------------------------------------------------------#

# Create first histogram
scatterplot_detailed_kripp_accuracy <- ggplot(data, aes(x = krippendorff_detailed, y = accuracy_detailed)) +
  geom_point(size = 3) +
  labs(title = "Detailed Labels", x = "Krippendorff's Alpha", y = "Accuracy") +
  theme_minimal()

# Create second histogram
scatterplot_simplified_kripp_accuracy <- ggplot(data, aes(x = krippendorff_simplified, y = accuracy_simplified)) +
  geom_point(size = 3) +
  labs(title = "Simplified Labels", x = "Krippendorff's Alpha", y = "Accuracy") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  scatterplot_detailed_kripp_accuracy, scatterplot_simplified_kripp_accuracy,
  ncol = 2,
  top = textGrob("Krippendorff's Alpha vs Accuracy", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)

#----------------------------------------------------------#
##### Scatterplot: Brennan-Prediger Alpha and Accuracy #####
#----------------------------------------------------------#

# Create first histogram
scatterplot_detailed_bp_accuracy <- ggplot(data, aes(x = bp_detailed, y = accuracy_detailed)) +
  geom_point(size = 3) +
  labs(title = "Detailed Labels", x = "Brennan-Prediger Alpha", y = "Accuracy") +
  theme_minimal()

# Create second histogram
scatterplot_simplified_bp_accuracy <- ggplot(data, aes(x = bp_simplified, y = accuracy_simplified)) +
  geom_point(size = 3) +
  labs(title = "Simplified Labels", x = "Brennan-Prediger Alpha", y = "Accuracy") +
  theme_minimal()

# Create combined plot with overall title and data source
grid.arrange(
  scatterplot_detailed_bp_accuracy, scatterplot_simplified_bp_accuracy,
  ncol = 2,
  top = textGrob("Brennan-Prediger Alpha vs Accuracy", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
)
