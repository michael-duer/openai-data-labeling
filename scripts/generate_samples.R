library(dplyr)

setwd("~/GitHub/openai-data-labeling/create testsets")

# Load file to create testsets from
data <- read.csv("TrainingData_manually_adjusted.csv",sep = ",")

#--------------------------#
#### Clean type2 values ####
#--------------------------#

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

#--------------------------#
#### Create sample sets ####
#--------------------------#

# Select 50 random rows
set.seed(42)  # Set a seed for reproducibility
random_sample_small <- clean_data %>%
  slice_sample(n=50) %>%
  select(-sentID)

# Export as csv file
write.csv(random_sample_small_2,"random_sample_small_2.csv", row.names = FALSE)

# Select 200 random rows
random_sample_long <- clean_data %>%
  slice_sample(n=200) %>%
  select(-sentID)

write.csv(random_sample_long,"random_sample_long.csv", row.names = FALSE)

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

#---------------------------------------#
##### Create 10 random sample sets ######
#---------------------------------------#

shuffled_data <- clean_data %>%
  slice_sample(prop=1)  %>% # Shuffle dataset
  select(-sentID)

# Split data into 10 non-overlapping samples
sample_sets <- split(shuffled_data, rep(1:10, each=200, length.out=nrow(shuffled_data)))

# Save each sample set as a CSV file
for (i in seq_along(sample_sets)) {
  write.csv(sample_sets[[i]], paste0("random_sample_", i, ".csv"))
}

print(sample_sets[1])

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
