library(tidyverse)
library(data.table)
args <- commandArgs(trailingOnly = TRUE)
BASE_DIR <- args[1]
file_path <- args[2]
dataset_name <- args[3]
item_index <- args[4]
lag <- as.numeric(args[5])
forecast_horizon <- as.numeric(args[6])

dataset <- readRDS(file.path(BASE_DIR, file_path))

dataset_long <- dataset %>%
  arrange(across(any_of(item_index))) %>%
  pivot_longer(cols = -all_of(item_index), names_to = "Date") %>%
  dplyr::select(any_of(item_index), value)

dataset_mean <- dataset_long %>%
  group_by(across(all_of(item_index))) %>%
  mutate(index = row_number()) %>%
  filter(index >= index[min(which(value > 0))]) %>%
  ungroup() %>%
  group_by(across(all_of(item_index))) %>%
  summarise(mean = mean(value)) %>%
  ungroup() %>%
  dplyr::select(any_of(item_index), mean)
write.csv(dataset_mean$mean, paste0(BASE_DIR, "/data/",dataset_name, "_series_means.csv"), row.names = FALSE)

dataset <- as.data.frame(dataset)
final_lags <- dataset[, c(1, ncol(dataset):(ncol(dataset)-lag+1))]
colnames(final_lags) <- c(item_index, sprintf("Lag%01d", 1:lag))
final_lags <- final_lags %>%
  left_join(dataset_mean, by = item_index) %>%
  arrange(across(any_of(item_index))) %>%
  group_by(across(all_of(item_index))) %>%
  ungroup() %>%
  mutate(across(c(2:(lag+1)), .fns = ~./mean)) %>%
  dplyr::select(starts_with("Lag"))
fwrite(final_lags, paste0(BASE_DIR, "/data/",dataset_name, "_final_lags.csv"))


dataset_long_dt <- data.table(dataset_long)
dataset_long_dt[, sprintf("Lag%01d", 1:lag) := shift(value, 1:lag, type = "lag"), by = item_index]
dataset_long_dt <- dataset_long_dt[dataset_long_dt[, -.I[1:lag], by = item_index]$V1]

# number of id_cols
id_cols <- length(item_index)
fourier_t <- c((forecast_horizon + ncol(dataset) - id_cols - lag) : (forecast_horizon + 1))
length(fourier_t)

# construct fourier features
fourier_features <- NULL
for (k in 1:4) {
  for (period in c(7, 364.25)) {
    feature_col <- rep(sin(2.0 * pi * k * fourier_t / period), nrow(dataset))
    fourier_features <- cbind(fourier_features, feature_col) 
    feature_col <- rep(cos(2.0 * pi * k * fourier_t / period), nrow(dataset))
    fourier_features <- cbind(fourier_features, feature_col)
  }
}

fourier_features <- as.data.frame(fourier_features)
colnames(fourier_features) <- paste0("fourier_features_", 1:16)

dataset_long_dt <- cbind(dataset_long_dt, fourier_features)

dataset_long_dt_embedded <- dataset_long_dt %>%
  left_join(dataset_mean, by = item_index) %>%
  mutate(y = value) %>%
  dplyr::select(y, starts_with("Lag"), mean, starts_with("fourier_features_")) %>%
  mutate(across(c(1:(lag+1)), .fns = ~./mean)) %>%
  dplyr::select(y, starts_with("Lag"), starts_with("fourier_features_"))

# dataset_long_dt_embedded <- dataset_long_dt_embedded %>%
#   dplyr::select(-starts_with("fourier_features_"))

saveRDS(dataset_long_dt_embedded, paste0(BASE_DIR, "/data/",dataset_name, "_embedded_matrix_fourier_terms_full.rds"))
