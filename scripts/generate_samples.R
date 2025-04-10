library(dplyr)

# This script cleans up the relationship categories in the TrainingData.csv file 
# and creates several samples based on that file that can be used as test sets to 
# test the labeling performance of the OpenAI API.

setwd("~/Documents/GitHub/openai-data-labeling/scripts")

# Load file to create testsets from
data <- read.csv("../data/TrainingData.csv",sep = ",")

#--------------------------#
#### Clean type2 values ####
#--------------------------#

# Standardize relationship categories (type2)

data_adjusted_types <- data %>%
  mutate(type2= case_when(
  type2 %in% c("negative 1","Negative 1","Negative1","negative1\n") ~ "negative1",
  type2 %in% c("Negative 2","negetive 2","negative 2") ~ "negative2",
  type2 %in% c("neutral 1","Neutral 1","nuetral 1","neutra1","Work for",
               "worked_for","work with","works for","Working for","work for",
               "working for","working_for") ~ "neutral1",
  type2 %in% c("neutral 2","Neutral 2","Netual 2","communication","communicate",
               "Communication","Communicate","commmunication","commuication",
               "communcation","commnuicate","communicate and meet","communictae",
               "meet and communicate","meet,communicate","meeting, and communicate",
               "met with","Meet","Meeting","meet","meeting") ~ "neutral2",
  type2 %in% c("positive 1","Positive 1","postive 1","postive1","Positive1",
               "potive1","positive1\n","positiven 1") ~ "positive1",
  type2 %in% c("positive 2","Positive 2","Positive2","postive 2","positiven 2",
               "positve2","positvie 2","relatives","Relatives","relative",
               "Others- person relation") ~ "positive2",
  TRUE ~ type2
  ))


# Remove sentences without head and tail as well as only keep certain types
allowed_types <- c("positive1","positive2","negative1","negative2","neutral1",
                   "neutral2","none")

clean_data <- data_adjusted_types %>%
  filter(!is.na(head) & !is.na(tail) & type2 %in% allowed_types) %>%
  select(sentID, sentence = sentence_fixed,head,tail,relation = type2)

#-----------------------#
#### Helper Function ####
#-----------------------#

set.seed(42)  # Set a seed for reproducibility

# Create and save non-overlapping random samples from a data frame
create_samples <- function(data, 
                           n_samples, 
                           sample_size, 
                           file_prefix = "random_sample_",
                           sample_index_offset = 0) {
  
  # Validate that input data is big enough
  rows_needed <- n_samples * sample_size
  if(nrow(data) < rows_needed) {
    stop("Not enough rows in input data to create the requested number of samples.")
  }  
  
  # Shuffle and truncate the dataset
  shuffled_data <- data %>%
    slice_sample(prop = 1) %>%
    slice(1:rows_needed)
  
  # Split into non-overlapping samples
  sample_sets <- split(shuffled_data, gl(n_samples, sample_size))
  
  # Save each sample to a csv file
  for (i in seq_along(sample_sets)) {
    # Use an offset value to avoid overwriting existing samples
    file_name <- paste0(file_prefix, i + sample_index_offset, ".csv") 
    write.csv(sample_sets[[i]], file_name, row.names = FALSE)
  }
}

#------------------------------------#
##### Create random sample sets ######
#------------------------------------#

# Create small sample sets (n=50)
create_samples(clean_data, 10, 50, file_prefix = "random_sample_small_", sample_index_offset = 2)

# Create large samples (n=500)
create_samples(clean_data, 2, 500, sample_index_offset = 10)

#------------------------------------------------------------#
#### Create samples from the beginning and end of dataset ####
#------------------------------------------------------------#

# Select 50 random rows from the first 200 rows
random_sample_beginning <- clean_data %>%
  head(200) %>%
  slice_sample(n=50) %>%
  select(-sentID)

write.csv(random_sample_beginning,"random_sample_beginning.csv", row.names = FALSE)

# Select 50 random rows from the last 200 rows
random_sample_end <- clean_data %>%
  tail(200) %>%
  slice_sample(n=50) %>%
  select(-sentID)

write.csv(random_sample_end,"random_sample_end.csv", row.names = FALSE)

#---------------------------------------------#
##### Create sample sets based on source ######
#---------------------------------------------#

# Define category filters
newspapers_ids <- c(37:62, 63:111, 946:1098, 1157:1185)
muller_report_ids <- c(112:849)
biographies_ids <- c(850:870, 871:925, 1099, 5218:5226)

# Filter data based on IDs
muller_data <- clean_data %>% filter(sentID %in% muller_report_ids)
muller_sample <- muller_data %>%
  slice_sample(n=200) %>%
  select(-sentID)

newspapers_data <- clean_data %>% filter(sentID %in% newspapers_ids)
newspaper_sample <- newspapers_data %>%
  slice_sample(n=200) %>%
  select(-sentID)

# As there are less than 200 sentences in bibliographies all are taken
biographies_data <- clean_data %>% filter(sentID %in% biographies_ids)

# Create CSV files
write.csv(muller_sample,"muller_sample.csv", row.names = FALSE)
write.csv(newspaper_sample,"newspapers_sample.csv", row.names = FALSE)
write.csv(biographies_data,"biographies_data.csv", row.names = FALSE)
