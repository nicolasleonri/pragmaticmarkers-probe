library(dplyr)
library(stringr)
library(tidyverse)
library(xtable)
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

# By aggregation and layers
df_aggregation_layer <- df_total %>%  group_by(aggregation, layer) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(-mean_accuracy)
xtable(df_aggregation_layer)

# By context
df_context <- df_total %>%  group_by(context) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(-mean_accuracy)
xtable(df_context)

# By tokenization
df_tokenization <- df_total %>%  group_by(tokenization) %>%  summarize(mean_accuracy = mean(accuracy, na.rm = FALSE)) %>% arrange(-mean_accuracy)
xtable(df_tokenization)

# By task
df_dc %>% arrange(-accuracy) %>% head()
df_tcomparison <- df_dc %>% group_by(source) %>%  summarize(mean_accuracy_t1 = mean(accuracy, na.rm = FALSE))

df_fr %>% arrange(-accuracy) %>% head()
df_tcomparison2 <- df_fr %>% group_by(source) %>%  summarize(mean_accuracy_t2 = mean(accuracy, na.rm = FALSE))
df_tcomparison <- merge(df_tcomparison, df_tcomparison2)

############################Fraser Task##################################
df <- read.csv2("results/frazer_predictions.csv")

predictions_df <- df %>%
  mutate(correct = rowSums(.[, -1] == PM), 
         incorrect = rowSums(.[, -1] != PM)) %>%
  select(Category, PM, correct, incorrect) %>%
  group_by(Category) %>%
  summarise(total_correct = sum(correct),
            total_incorrect = sum(incorrect),
            percentage = (total_correct/(total_correct+total_incorrect))*100,
            .groups = 'drop')  


