library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)

setwd("~/GitHub/openai-data-labeling/benchmark output")

# Load file to create testsets from
data <- read.csv("evaluation.csv",sep = ",")

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
  top = textGrob("Accuracy distribution", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
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
  top = textGrob("Krippendorff's Alpha distribution", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
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
  top = textGrob("Brennan-Prediger Alpha distribution", gp = gpar(fontsize = 16, fontface = "bold")),
  bottom = textGrob("Based on 8 different samples containing a total of 1'985 sentences.", gp = gpar(fontsize = 10, fontface = "italic"))
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
