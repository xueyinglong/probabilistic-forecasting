# Creating embedded matrix and final lags to train the global models for a given lag
create_input_matrix <- function(dataset, lag, step, remove_leading_zeros = TRUE, price_data = NULL, price_test_data = NULL){
  embedded_series <- NULL
  final_lags <- NULL
  series_means <- NULL
  
  for (i in 1:nrow(dataset)) {
    print(i)
    time_series <- as.numeric(dataset[i,])
    
    if(remove_leading_zeros)
      time_series <- leadtrail(time_series, rm = "zeros", lead = TRUE, trail = FALSE)
    
    if (length(time_series) < lag+1)
      time_series <- c(rep(0, lag+1-length(time_series)), time_series)
    
    if(!is.null(price_data)){
      price_series <- as.numeric(price_data[i,])
      
      if(remove_leading_zeros)  
        price_series <- leadtrail(price_series, rm = "zeros", lead = TRUE, trail = FALSE) 
    }    
    
    mean <- mean(time_series)
    
    if(mean != 0){
      # Mean normalisation
      time_series <- time_series / mean
    }else
      mean <- 1

    series_means <- c(series_means, mean)
    
    # Embed the series
    if(!is.null(step) & step != 1){
      embedded <- embed(time_series, lag + step)[,-c(2:step)]
    }else
      embedded <- embed(time_series, lag + 1)
      
    if(!is.null(price_data))
      embedded <- cbind(embedded, price_series[(lag + 1):length(price_series)])
    
    if (!is.null(embedded_series)) 
      embedded_series <- as.matrix(embedded_series)
    
    embedded_series <- rbind(embedded_series, embedded)
    
    # Creating the test set
    if (!is.null(final_lags)) 
      final_lags <- as.matrix(final_lags)
    
    current_series_final_lags <- t(as.matrix(rev(tail(time_series, lag))))
    
    if(!is.null(price_data))  
      current_series_final_lags <- cbind(current_series_final_lags, price_test_data[i, 1])
    
    final_lags <- rbind(final_lags, current_series_final_lags)
  }
  
  # Adding proper column names for embedded_series and final_lags
  embedded_series <- as.data.frame(embedded_series)
  colnames(embedded_series)[1] <- "y"
  colnames(embedded_series)[2:(lag + 1)] <- paste("Lag", 1:lag, sep = "")
  
  final_lags <- as.data.frame(final_lags)
  colnames(final_lags)[1:lag] <- paste("Lag", 1:lag, sep = "")
  
  if(!is.null(price_data)){
    colnames(embedded_series)[(lag + 2)] <- "price"
    colnames(final_lags)[(lag + 1)] <- "price"
  }  
  
  list(embedded_series, final_lags, series_means)
}


# Randomly choosing series from a given dataset
sample_series <- function(dataset, instance_number, seed = 1){
  set.seed(seed)
  required_series <- sample(1:nrow(dataset), instance_number, replace = FALSE)
  output_data <- dataset[required_series,]
  output_data[,-2]
}

# Creating embedded matrix and save them using data.table package
# Creating embedded matrix and final lags to train the global models for a given lag
# @item_index the col name in the df that contains the unique index for each series
create_presaved_input_matrix <- function(dataset_name, BASE_DIR, file_path, item_index, lag, forecast_horizon) {
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
    group_by(across(all_of(item_index))) %>%
    mutate(index = row_number()) %>%
    filter(index >= index[min(which(value > 0))]) %>%
    ungroup() %>%
    mutate(y = value) %>%
    dplyr::select(y, starts_with("Lag"), mean, starts_with("fourier_features_")) %>%
    mutate(across(c(1:(lag+1)), .fns = ~./mean)) %>%
    dplyr::select(y, starts_with("Lag"), starts_with("fourier_features_"))
  
  fwrite(dataset_long_dt_embedded, paste0(BASE_DIR, "/data/",dataset_name, "_embedded_matrix_fourier_terms.rds"))
}
