library(dplyr)
library(stringr)
library(tidyverse)
library(xtable)
library(data.table)
############################Randomized baseline#################################
# Discourse connective
df <- read.csv2("data/discourse_connective/probing_data.csv")
samples <- length(df$Before)
n_markers <- length(unique(df$PM))
n_distribution <- df %>% group_by(PM) %>% count()
random_baseline_accuracy <- 1 / n_markers
expected_correct_predictions <- random_baseline_accuracy * samples
cat("Randomized Baseline Accuracy:", random_baseline_accuracy, "\n")
cat("Expected Number of Correct Predictions (Random Baseline):", expected_correct_predictions, "\n")
cat("Total Number of Samples:", samples, "\n")
cat("Number of Unique Markers:", n_markers, "\n")

# Discourse connective
df <- read.csv2("data/frazer_categorization/probing_data_2.csv")
samples <- length(df$Before)
n_markers <- length(unique(df$PM))
n_distribution <- df %>% group_by(PM) %>% count()
random_baseline_accuracy <- 1 / n_markers
expected_correct_predictions <- random_baseline_accuracy * samples
cat("Randomized Baseline Accuracy:", random_baseline_accuracy, "\n")
cat("Expected Number of Correct Predictions (Random Baseline):", expected_correct_predictions, "\n")
cat("Total Number of Samples:", samples, "\n")
cat("Number of Unique Markers:", n_markers, "\n")

total_randomized_baseline <- (1 / 14) * 500 + (1/25) * 226
total_randomized_baseline <- (((1 / 14) * 500 + (1 / 25) * 226) / (500 + 226))*100
############################General results##################################
# Pre-processing
preprocess <- function(df) {
  df$accuracy <- as.numeric(df$accuracy)
  
  df <- df %>%
    mutate(split_col = str_split(experiment, "/")) %>%
    mutate(
      context = map_chr(split_col, 1),
      language = map_chr(split_col, 2),
      model = map_chr(split_col, 3),
      tokenization = map_chr(split_col, 4),
      layer = map_chr(split_col, 5),
      aggregation = map_chr(split_col, 6)
    ) %>%
    select(-split_col) # remove temporary split column
  
  df <- df %>%
    mutate(
      source = case_when(
        language == "en" & model == "multi" ~ "mBert_base",
        language == "en" & model == "mono" ~ "Bert_base",
        language == "en2" & model == "mono" ~ "Bert_large",
        TRUE ~ NA_character_  # Or any default value if none of the conditions match
      )
    )
  return(df)
}

df_dc <- preprocess(read.csv2("results/discourse_connective_results.csv"))
df_fr <- preprocess(read.csv2("results/frazer_results.csv"))
df_total <- rbind(df_dc, df_fr)
df_fr <- df_fr[,c(12,6,9,11,10,5,4)]
df_fr$layer <- as.numeric(df_fr$layer)
df_fr <- df_fr %>% arrange(source, context, tokenization, aggregation, layer)
print(xtable(df_fr), include.rownames=FALSE)

# General results
df_gr <- df_total %>%  group_by(source) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(-mean_accuracy)
print(xtable(df_gr), include.rownames=FALSE)

# By aggregation
df_aggregation <- df_total %>%  group_by(source, aggregation) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(source, -mean_accuracy)
print(xtable(df_aggregation), include.rownames=FALSE)

# By aggregation and layers
df_aggregation_layer <- df_total %>% group_by(source, aggregation, layer) %>%
  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(source, desc(mean_accuracy)) %>%
  slice_max(mean_accuracy, n = 3, with_ties = FALSE) %>% ungroup()
print(xtable(df_aggregation_layer), include.rownames=FALSE)

# By context
df_context <- df_total %>%  group_by(source, context) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(source, -mean_accuracy)
print(xtable(df_context), include.rownames=FALSE)

# By tokenization
df_tokenization <- df_total %>%  group_by(source, tokenization) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(source, -mean_accuracy)
print(xtable(df_tokenization), include.rownames=FALSE)

# By task
df_tcomparison <- df_dc %>% group_by(source) %>%  summarize(mean_accuracy_t1 = mean(accuracy, na.rm = FALSE))
df_tcomparison2 <- df_fr %>% group_by(source) %>%  summarize(mean_accuracy_t2 = mean(accuracy, na.rm = FALSE))
df_tcomparison <- merge(df_tcomparison, df_tcomparison2)
print(xtable(df_tcomparison), include.rownames=FALSE)

############################Fraser Task##################################
df <- read.csv2("results/frazer_predictions.csv")

# In general
predictions_df <- df %>%
  mutate(correct = rowSums(.[, -1] == PM, na.rm = TRUE), 
         incorrect = rowSums(.[, -1] != PM,na.rm = TRUE)) %>%
  select(Category, PM, correct, incorrect) %>%
  group_by(Category) %>%
  summarise(total_correct = sum(correct),
            total_incorrect = sum(incorrect),
            percentage = (total_correct/(total_correct+total_incorrect))*100,
            .groups = 'drop')
print(xtable(predictions_df), include.rownames=FALSE)

# By configuration
df_config <- df[4:length(colnames(df))]
df_config$PM <- df$PM

df_config <- df_config %>%
  mutate(across(2:(ncol(df_config)-1), ~ if_else(. == PM, 1, 0)))

df_config <- df_config[1:(length(df_config)-1)]

unique_sum <- df_config %>% group_by(Category) %>%
  summarize(total = n()) 

df_config <- df_config %>% group_by(Category) %>% summarise_all(sum)

t_df_config <- data.frame(t(df_config[-1]))
colnames(t_df_config) <- df_config$Category

t_df_config <- t_df_config %>%
  mutate(
    basic_total = unique_sum$total[unique_sum$Category == "basic"],
    commentary_total = unique_sum$total[unique_sum$Category == "commentary"],
    discourse_total = unique_sum$total[unique_sum$Category == "discourse"],
    parallel_total = unique_sum$total[unique_sum$Category == "parallel"]
  ) %>%
  # Calculate the percentage of correct predictions
  mutate(
    basic_percentage = (basic / basic_total) * 100,
    commentary_percentage = (commentary / commentary_total) * 100,
    discourse_percentage = (discourse / discourse_total) * 100,
    parallel_percentage = (parallel / parallel_total) * 100
  )

summary(t_df_config)
t_df_config$experiment = rownames(t_df_config)
rownames(t_df_config) <- NULL

preprocess2 <- function(df) {
  df <-  df %>%
    mutate(split_col = str_split(experiment, "\\.")) %>%
    mutate(
      context = map_chr(split_col, 1),
      language = map_chr(split_col, 2),
      model = map_chr(split_col, 3),
      tokenization = map_chr(split_col, 4),
      layer = map_chr(split_col, 5),
      aggregation = map_chr(split_col, 6)
    ) %>%
    select(-split_col)
  
  df <- df %>%
    mutate(
      source = case_when(
        language == "en" & model == "multi" ~ "mBert_base",
        language == "en" & model == "mono" ~ "Bert_base",
        language == "en2" & model == "mono" ~ "Bert_large",
        TRUE ~ NA_character_  # Or any default value if none of the conditions match
      )
    )
  return(df)
}

df_config <- preprocess2(t_df_config)

# By context and tokenization
df_config_context_tokenization <- df_config %>% group_by(context, tokenization) %>% summarize(
  basic_accuracy = mean(basic_percentage, na.rm = TRUE),
  commentary_accuracy = mean(commentary_percentage, na.rm = TRUE),
  parallel_accuracy = mean(parallel_percentage, na.rm = TRUE),
  discourse_accuracy = mean(discourse_percentage, na.rm = TRUE))
print(xtable(df_config_context_tokenization), include.rownames=FALSE)

# By source and context
df_config_source_context <- df_config %>% group_by(source, context) %>% summarize(
  basic_accuracy = mean(basic_percentage, na.rm = TRUE),
  commentary_accuracy = mean(commentary_percentage, na.rm = TRUE),
  parallel_accuracy = mean(parallel_percentage, na.rm = TRUE),
  discourse_accuracy = mean(discourse_percentage, na.rm = TRUE))
print(xtable(df_config_source_context), include.rownames=FALSE)

# By source and context
df_config_context_aggregation <- df_config %>% group_by(context, aggregation) %>% summarize(
  basic_accuracy = mean(basic_percentage, na.rm = TRUE),
  commentary_accuracy = mean(commentary_percentage, na.rm = TRUE),
  parallel_accuracy = mean(parallel_percentage, na.rm = TRUE),
  discourse_accuracy = mean(discourse_percentage, na.rm = TRUE))
print(xtable(df_config_context_aggregation), include.rownames=FALSE)
