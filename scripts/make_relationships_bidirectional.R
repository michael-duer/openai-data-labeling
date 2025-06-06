library(dplyr)
library(stringr)

# This script adjusts the sample data by establishing bidirectional relationships. After applying the script to a file, the "relation" column will be replaced by two columns: "rel_head_tail" and "rel_tail_head". The labels will also change from positive1, positive2, neutral1, neutral2, etc. to positive/neutral/negative/none.

setwd("~/Documents/GitHub/openai-data-labeling/scripts")

# Load file to apply changes to
data <- read.csv("../data/random_sample_small_1.csv",sep = ",")

# Adjust columns
data_adjusted <- data %>%
    rename(relation = "rel_head_tail") %>%
    mutate(
        # Add relationship in other direction by either copying the relationship (if mutual) or add "none" (if one-sided)
        "rel_tail_head" = "rel_head_tail" %>% case_when(
            # If last char = 1 (one-sided) -> set relation to none
            str_sub("rel_head_tail",-1) == "1" ~ "none"
            # If last char = 2 (mutual) -> keep relation and remove number
            str_sub("rel_head_tail",-1) == "2" ~ sub("2","","rel_head_tail")
            # If last char of string is neither (probably none) -> keep relation
            TRUE ~ "rel_head_tail")
        # Remove number from relationship type
        "rel_head_tail" = gsub("[1-2]","","rel_head_tail"))


# Export file with adjusted name