library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)

# This script creates different plots to asses and compare the performance of 
# different models and prompts on different/the same samples.

setwd("~/Documents/GitHub/openai-data-labeling/scripts")

# Load file to create testsets from
data <- read.csv("../data/evaluation/evaluation.csv",sep = ",")

# TODO: Remove after testing
sample1 <- read.csv("../data/samples/random_sample_1.csv",sep = ",")
output_sample1 <- read.csv("../data/api_output/detailed_output_gpt-4.5-preview-2025-02-27_new_prompt2_random_sample_1.csv",sep = ",")

# Compare sample
prompt_comparison <- c("new_prompt2.txt","structured_prompt.txt")
sample_small1 <- data %>%
  filter(dataset=="random_sample_small_1.csv") %>%
  #filter(prompt %in% prompt_comparison) %>%
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
