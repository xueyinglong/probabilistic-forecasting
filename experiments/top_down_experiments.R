# The top down forecasting framework
# Step 1. Training standard machine learning models 
# such as LightGBM, linear regression models globally
# Models are trained with aggregated series
# Implemented with the M5 datasets
# train on level 10 -> forecast on level 12
# M5 data: https://github.com/Mcompetitions/M5-methods/blob/master/README.md


library(tidyverse)
library(tsutils)
library(data.table)
library(dplyr)

set.seed(0)

BASE_DIR <- "~/probabilistic-forecasting"

source(file.path(BASE_DIR, "utils", "global_model_helper.R"))
source(file.path(BASE_DIR, "utils", "global_models.R"))
source(file.path(BASE_DIR, "utils", "global_forecasting.R"))
source(file.path(BASE_DIR, "utils", "wrappers.R"))

# make sure series are aligned (arranged by id)
M5_level_10 <- readRDS(paste0(BASE_DIR, "/data/M5_level_10.rds")) %>% arrange(item_id)

# run only once
# create presaved input matrix
create_presaved_input_matrix("M5_level_10", BASE_DIR, "/data/M5_level_10.rds", c("item_id"), lag = 100, forecast_horizon = 28)

# test a single method
methods <- c("pooled_regression")
# test more methods...
# methods <- c("lightgbm_tweedie", "lightgbm_poisson", "glmnet", "pooled_regression")

lag <- 100
use_quad_terms <- FALSE
dataset_name <- "M5_level_10"
presaved_file_path <- list("embedded_matrix" = paste0(BASE_DIR, "/data/",dataset_name, "_embedded_matrix_fourier_terms.rds"),
                           "final_lags" = paste0(BASE_DIR, "/data/",dataset_name, "_final_lags.csv"),
                           "mean_list" = paste0(BASE_DIR, "/data/",dataset_name, "_series_means.csv"))


# ######## forecast all level 10 data as a whole
for (method in methods) {
  do_fixed_horizon_global_forecasting(dataset_name = "M5_level_10", method = ifelse(grepl("lightgbm", method), "lightgbm", method), lag = lag, step = 1, dataset = M5_level_10,
                                      loss = ifelse(grepl("lightgbm", method), sub("lightgbm_", "", method), "none"), direct = FALSE, forecast_horizon = 28,
                                      full_price_dataset = NULL, run_with_prices = FALSE, percentile = NULL, use_quad_terms = use_quad_terms, include_real_vals = FALSE, 
                                      use_presaved_matrix = TRUE, presaved_file_path = presaved_file_path, with_fourier = T, fourier_periods = c(7, 365.25))
}
