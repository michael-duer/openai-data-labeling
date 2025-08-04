library(dplyr)
library(stringr)

# This script adjusts the relationships in all files inside the "../data/samples/" 
# folder to be bidirectional. The "relation" column will be replaced by the two 
# columns: "rel_head_tail" and "rel_tail_head". The labels will also change from 
# positive1,positive2,neutral1,neutral2, etc. to positive/neutral/negative/none.

setwd("~/Documents/GitHub/openai-data-labeling/scripts")

# Helper Function to adjust columns and relationships
adjust_relationships <- function(data){
  data %>%
    rename(rel_head_tail = relation) %>%
    mutate(
      # Add relationship in other direction by either copying the relationship 
      # (if mutual) or add "none" (if one-sided)
      rel_tail_head = case_when(
        # If one-sided -> set relation to none
        str_sub(rel_head_tail,-1) == "1" ~ "none",
        # If mutual -> keep relation and remove number
        str_sub(rel_head_tail,-1) == "2" ~ sub("2","",rel_head_tail),
        # If last char of string is neither (probably none) -> keep relation
        TRUE ~ rel_head_tail
      ),
      # Remove number from relationship type
      rel_head_tail = gsub("[1-2]","",rel_head_tail)
    )
}

# Read all files in "samples" folder
files <- list.files(path = "../data/samples", pattern = "*.csv", full.names = TRUE)

# Apply transformation to all files
for (file in files) {
  # Exclude already transformed files
  if (file %in% c("../data/samples/random_sample_small_1 bidirectional.csv",
                  "../data/samples/random_sample_small_5_bidirectional.csv")) next
  
  tryCatch(
    {
      data <- read.csv(file,sep = ",")
      transformed_data <- adjust_relationships(data)
      # Save with same name to output directory
      filename <- basename(file)
      write.csv(transformed_data, file=paste0("../data/samples/new/",filename), row.names = FALSE)
    },
    # Print error message
    error=function(e) {
      message(paste0("An Error occurred while processing ",file))
      print(e)
    },
    # Print warning message
    warning=function(w) {
      message(paste0("A Warning occurred while processing ",file))
      print(w)
    }
  )
}