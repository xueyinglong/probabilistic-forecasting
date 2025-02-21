do_fixed_horizon_global_forecasting <- function(dataset_name, method, lag, step = NULL, dataset, remove_leading_zeros = TRUE, integer_conversion = FALSE, loss = "tweedie", direct = FALSE, percentile = NULL, forecast_horizon = 28, full_price_dataset = NULL, run_with_prices = FALSE, use_quad_terms = FALSE, include_real_vals = TRUE, use_presaved_matrix = FALSE, presaved_file_path = NULL, with_fourier = F, fourier_periods = c(7, 365.25)){
  # print(as.list(environment()))
  catalog_ids <- dataset[[1]]
  dataset <- dataset[,-1]
  if (include_real_vals) {
    train_series <- as.data.frame(dataset[,1:(ncol(dataset) - forecast_horizon)])
    actual_matrix <- as.data.frame(dataset[,(ncol(dataset) - forecast_horizon + 1) : ncol(dataset)])
  } else {
    train_series <- as.data.frame(dataset)
  }
  
  if(!is.null(full_price_dataset)){
    price_data <- as.data.frame(full_price_dataset[,1:(ncol(full_price_dataset) - forecast_horizon)])
    price_test_data <- as.data.frame(full_price_dataset[,(ncol(full_price_dataset) - forecast_horizon + 1) : ncol(full_price_dataset)])
  }else
    price_test_data <- NULL  
  
  start_time <- Sys.time()
  
  print("Started Forecasting")
  print(start_time)
  
  # Forecasting
  if(run_with_prices & !is.null(full_price_dataset)){  
    # Run with past lags and prices
    forecast_matrix <- start_forecasting(BASE_DIR, dataset_name, train_series, lag, step, forecast_horizon, method, remove_leading_zeros, loss, direct, percentile, price_data, price_test_data, run_with_prices, use_quad_terms = use_quad_terms, use_presaved_matrix = use_presaved_matrix, presaved_file_path = presaved_file_path, with_fourier = with_fourier, fourier_periods = fourier_periods) 
  }else{
    # Run only with past lags
    forecast_matrix <- start_forecasting(BASE_DIR, dataset_name, train_series, lag, step, forecast_horizon, method, remove_leading_zeros, loss, direct, percentile, run_with_prices = FALSE, use_quad_terms = use_quad_terms, use_presaved_matrix = use_presaved_matrix, presaved_file_path = presaved_file_path, with_fourier = with_fourier, fourier_periods = fourier_periods) 
  }
  
  forecast_matrix[[2]][is.na(forecast_matrix[[2]])] <- 0
  forecast_matrix[[2]][forecast_matrix[[2]] < 0] <- 0
  
  if(integer_conversion)
    forecast_matrix[[2]] <- round(forecast_matrix[[2]])
  
  if(method == "lightgbm" & !is.null(percentile))
    method <- paste0(method, "_", loss, "_", percentile, "_recursive_norm_pfcast")
  
  if(method == "lightgbm" & is.null(percentile))
    method <- paste0(method, "_", loss)
  
  if(!is.null(step) & !is.null(percentile) & direct == TRUE){
    direct_step <- paste0("_direct_", step, "_step")
  }else{
    direct_step <- NULL
  }
  
  quad_text <- ""
  quad_text <- ifelse(use_quad_terms, "_quadratic", "")
  file_name <- paste0(dataset_name, direct_step, "_", method, quad_text, "_lag_", lag)
  
  if(run_with_prices & !is.null(full_price_dataset))  
    file_name <- paste0(file_name, "_with_prices")
  
  write.table(cbind(catalog_ids, forecast_matrix[[2]]), file.path(BASE_DIR, "results", "forecasts", paste0(file_name, ".txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE)
  write.table(cbind(catalog_ids, forecast_matrix[[1]]), file.path(BASE_DIR, "results", "forecasts", paste0(file_name, "_normalised", ".txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE)
  
  end_time <- Sys.time()
  
  print("Finished Forecasting")
  
  # Execution time
  exec_time <- end_time - start_time
  print(exec_time)
  write(paste(exec_time, attr(exec_time, "units")), file = file.path(BASE_DIR, "results", "execution_times", paste0(file_name, ".txt"), fsep = "/"), append = FALSE)
  
  # Error calculations
  # TODO: Turn off error calculation if loss = "quantile"
  # if(loss != "quantile"){
  #   forecast_matrix <- as.matrix(forecast_matrix[[2]])
  #   calculate_errors(forecast_matrix, actual_matrix, train_series, seasonality, file.path(BASE_DIR, "results", "errors", file_name, fsep = "/"), remove_leading_zeros, price_test_data)
  # }
  forecast_matrix <- as.matrix(forecast_matrix[[2]])
  # calculate_errors(forecast_matrix, actual_matrix, train_series, seasonality, file.path(BASE_DIR, "results", "errors", file_name, fsep = "/"), remove_leading_zeros, price_test_data)
}