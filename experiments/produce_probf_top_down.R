# The top down forecasting framework
# Step 2. Produce probabilistic forecasts for bottom level
# Disaggregate point forecasts based on historical proportion
# Probabilistic forecasts are produced based on assumptions
# i.e., negative binomial distribution and Poisson distribution
# Implemented with the M5 datasets
# train on level 10 -> forecast on level 12
# https://github.com/Mcompetitions/M5-methods/blob/master/README.md

library(tidyverse)
library(data.table)

# ######## get forecasts ########
forecast_horizon <- 28
quantiles <- c(0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995)
test_set <- read.csv("./data/sales_test_evaluation.csv")
test_set <- test_set[,c(1, (ncol(test_set) - forecast_horizon + 1):ncol(test_set))]
lag <- 100

# methods <- c("lightgbm_tweedie", "lightgbm_poisson", "lightgbm_negbin", "glmnet", "pooled_regression")
methods <- c("pooled_regression")

# method <- "lightgbm_tweedie"
for (method in methods) {
  if (method == "lightgbm_negbin")
    forecasts <- read.csv(paste0("results/forecasts/M5_level_10_", method, "_lag_", lag, ".csv"), header = F)
  else
    forecasts <- read.table(paste0("results/forecasts/M5_level_10_", method, "_lag_", lag, ".txt"), sep = ",")
  colnames(forecasts) <- colnames(test_set)
  
  forecasts <- forecasts %>%
    pivot_longer(
      cols = starts_with("d_"),
      names_to = "date", 
      values_to = "forecast"
    )
  
  # get quantiles when assumed to be Poisson distribution
  top_down_fcast_poisson_quantiles <- forecasts %>%
    left_join(readRDS("data/M5_historic_prop.rds"), by = "item_id") %>%
    mutate(product_forecast = forecast*prop_product_id_total_daily_quantity) %>%
    select(item_id, store_id, date, product_forecast)
  
  for (quantile in quantiles) {
    col_name <- paste(format(round(quantile, 3), nsmall = 3), "evaluation", sep = "_")
    top_down_fcast_poisson_quantiles[[col_name]] = qpois(quantile, top_down_fcast_poisson_quantiles$product_forecast)
  }
  
  top_down_fcast_poisson_quantiles <- top_down_fcast_poisson_quantiles %>%
    select(item_id, store_id, date, ends_with("_evaluation")) %>%
    pivot_longer(
      cols = ends_with("_evaluation"),
      names_to = "quantiles", 
      values_to = "quantile_forecast"
    ) %>%
    pivot_wider(
      names_from = date,
      id_cols = c(item_id, store_id, quantiles),
      values_from = `quantile_forecast`
    ) %>%
    mutate(id = paste(item_id, store_id, quantiles, sep = "_"), .before = 1) %>%
    select(id, starts_with("d_"))
  
  colnames(top_down_fcast_poisson_quantiles) <- sprintf("F%01d", 1:28)
  
  write.csv(top_down_fcast_poisson_quantiles, paste0("M5/submission_", method, "_level12_poisson_lag_", lag, "_", as.numeric(as.POSIXct(Sys.time())), ".csv"), row.names = F)
  
  
  # get quantiles when assumed to be Negative Binomial distribution
  top_down_fcast_negbin_quantiles <- forecasts %>%
    left_join(readRDS("data/M5_historic_prop.rds"), by = "item_id") %>%
    mutate(product_forecast = forecast*prop_product_id_total_daily_quantity,
           p_hat = product_forecast/var_hat,
           n_hat = product_forecast*p_hat/(1-p_hat)
           ) %>%
    select(item_id, store_id, date, product_forecast, p_hat, n_hat)
  
  for (quantile in quantiles) {
    col_name <- paste(format(round(quantile, 3), nsmall = 3), "evaluation", sep = "_")
    top_down_fcast_negbin_quantiles[[col_name]] = ifelse(top_down_fcast_negbin_quantiles$p_hat >= 1, qpois(quantile, top_down_fcast_negbin_quantiles$product_forecast), qnbinom(quantile, size = top_down_fcast_negbin_quantiles$n_hat, prob = top_down_fcast_negbin_quantiles$p_hat))
  }
  
  top_down_fcast_negbin_quantiles <- top_down_fcast_negbin_quantiles %>%
    select(item_id, store_id, date, ends_with("_evaluation")) %>%
    pivot_longer(
      cols = ends_with("_evaluation"),
      names_to = "quantiles", 
      values_to = "quantile_forecast"
    ) %>%
    pivot_wider(
      names_from = date,
      id_cols = c(item_id, store_id, quantiles),
      values_from = `quantile_forecast`
    ) %>%
    mutate(id = paste(item_id, store_id, quantiles, sep = "_"), .before = 1) %>%
    select(id, starts_with("d_"))
  
  colnames(top_down_fcast_negbin_quantiles) <- sprintf("F%01d", 1:28)
  
  write.csv(top_down_fcast_negbin_quantiles, paste0("M5/submission_", method, "_level12_negbin_lag_", lag, "_", as.numeric(as.POSIXct(Sys.time())), ".csv"), row.names = F)
  
}
