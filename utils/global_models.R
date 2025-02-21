# Implementation of global models

library(glmnet)
library(lightgbm)
library(Matrix)
library(mgcv)
library(MASS)

set.seed(1)


# Forecasting with different lags
start_forecasting <- function(BASE_DIR, dataset_name, dataset, lag, step, forecast_horizon, model_type = "pooled_regression", remove_leading_zeros = TRUE, loss = "tweedie", direct = FALSE, percentile, price_data = NULL, price_test_data = NULL, run_with_prices, use_quad_terms, use_presaved_matrix = FALSE, presaved_file_path = NULL, with_fourier = F, fourier_periods = c(7, 365.25)){
  # print(as.list(environment()))
  if (use_presaved_matrix) {
    result <- tryCatch({
      embedded_series <- as.data.frame(fread(presaved_file_path$embedded_matrix))
      final_lags <- as.data.frame(fread(presaved_file_path$final_lags))
      series_means <- read.csv(presaved_file_path$mean_list)[,1]
      list(embedded_series, final_lags, series_means)
    }, error = function(e) {
      stop(e)
    })
    print(dataset_name)
    embedded_series <- result[[1]] # Embedded matrix
    final_lags <- result[[2]] # Test set
    series_means <- result[[3]] # Mean value of each series
    
  } else {
    
    # Creating embedded matrix (for model training) and test set
    result <- create_input_matrix(dataset, lag, step, remove_leading_zeros, price_data, price_test_data)
    
    embedded_series <- result[[1]] # Embedded matrix
    final_lags <- result[[2]] # Test set
    series_means <- result[[3]] # Mean value of each series
      
  }
  
  fit_model(BASE_DIR, dataset_name, embedded_series, lag, step, final_lags, forecast_horizon, series_means, model_type, loss, direct, percentile, price_test_data, run_with_prices, use_quad_terms, with_fourier, fourier_periods)
}


# Fit and forecast from a global model
fit_model <- function(BASE_DIR, dataset_name, fitting_data, lag, step, final_lags, forecast_horizon, series_means, model_type = "pooled_regression", loss = "tweedie", direct = FALSE, percentile, price_test_data = NULL, run_with_prices, use_quad_terms, with_fourier, fourier_periods) {
  # print(as.list(environment()))
  # Create the formula
  formula <- "y ~ "
  for(predictor in 2:ncol(fitting_data)){
    if(predictor != ncol(fitting_data)){
      if (model_type == "gam") 
        formula <- paste0(formula, "s(", colnames(fitting_data)[predictor], ") + ")
      else
        formula <- paste0(formula, colnames(fitting_data)[predictor], " + ")
      
      if (use_quad_terms) {
        formula <- paste0(formula, "I(", colnames(fitting_data)[predictor], "^2) + ")
      }
    } else{
      if (model_type == "gam")
        formula <- paste0(formula, "s(", colnames(fitting_data)[predictor], ")")
      else
        formula <- paste0(formula, colnames(fitting_data)[predictor])
      
      if (use_quad_terms) {
        formula <- paste0(formula, "+ I(", colnames(fitting_data)[predictor], "^2)")
      }
    }
  }
  if (model_type != "glmnet")
    formula <- paste(formula, "+ 0", sep="")
  formula <- as.formula(formula)
  # print(formula)
  if(run_with_prices)  
    dataset_name <- paste0(dataset_name, "_with_prices")  
  
  # Fit a global model 
  quad_text <- ""
  if(model_type == "pooled_regression") {
    model <- glm(formula = formula, data = fitting_data)
    # Save model  
    # saveRDS(model, paste0(BASE_DIR, "/results/models/", dataset_name, "_pooled_regression_", ".rds"))
  } else if(model_type == "glm_poisson") {
    model <- glm(formula = formula, data = fitting_data, family = "poisson")
    # Save model  
    quad_text <- ifelse(use_quad_terms, "_quadratic", "")
    saveRDS(model, paste0(BASE_DIR, "/results/models/", dataset_name, "_glm_poisson", quad_text,".rds"))
  } else if(model_type == "glm_negbin") {
    quad_text <- ifelse(use_quad_terms, "_quadratic", "")
    model <- glm.nb(formula = formula, data = fitting_data)
    # Save model  
    saveRDS(model, paste0(BASE_DIR, "/results/models/", dataset_name, "_glm_negbin", quad_text, ".rds"))
  } else if(model_type == "gam") {
    model <- gam(formula = formula, data = fitting_data, method = "REML", family=nb(link="log"))
    # Save model  
    # saveRDS(model, paste0(BASE_DIR, "/results/models/", dataset_name, "_gam", ".rds"))
  } else if(model_type == "glmnet") {
    model <- cv.glmnet.f(formula = formula, data = fitting_data, alpha = 1)
    # Save model  
    # saveRDS(model, paste0(BASE_DIR, "/results/models/", dataset_name, "_glmnet", ".rds"))
  } else if(model_type == "lightgbm"){
    # Defining hyperparameters
    
    if(loss == "quantile") {
      lgb.grid <- list(objective = loss,
                       alpha = percentile/100,
                       bagging_freq = 1,
                       lambda_l2 = 0.1,
                       learning_rate = 0.075,
                       num_leaves = 128,
                       min_data_in_leaf = 100,
                       boosting_type = 'gbdt',
                       force_row_wise = TRUE,
                       sub_row = 0.75,
                       verbosity = -1,
                       num_iterations = 1200
      )
    } else {
      lgb.grid <- list(objective = loss,
                       # linear_tree = TRUE,
                       metric = "rmse",
                       #bagging_freq = 1,
                       #lambda_l2 = 0.1,
                       #learning_rate = 0.075,
                       #num_leaves = 128,
                       #min_data_in_leaf = 100,
                       boosting_type = 'gbdt',
                       #force_row_wise = TRUE,
                       #sub_row = 0.75,
                       verbosity = -1,
                       #num_iterations = 1200
                       num_iterations = 100
      )
    }
    
    train <- Matrix(as.matrix(fitting_data[-1]), sparse = TRUE)
    y_train <- as.numeric((fitting_data[,1]))
    dtrain <- lgb.Dataset(data = train, label = y_train, free_raw_data = FALSE)
    
    # Fit the model
    # model <- lgb.train(params = lgb.grid, data = dtrain)
    model <- lgb.train(params = lgb.grid, data = dtrain)
    
    # Save model
    if(loss == "quantile" & !is.null(step)){
      lgb.save(model, filename = paste0(BASE_DIR, "/results/models/", dataset_name, "_direct_", step, "_step", "_lightgbm_", loss, "_", percentile, ".txt"))
    } else {
      lgb.save(model, filename = paste0(BASE_DIR, "/results/models/", dataset_name, "_lightgbm_", loss, ".txt"))
    } 
    # model <- lgb.train(params = lgb.grid, data = dtrain, categorical_feature = categoricals.vec )
  }
  
  # Do forecasting
  forec_recursive(BASE_DIR, dataset_name, lag, model, final_lags, forecast_horizon, series_means, model_type, loss, direct, price_test_data, with_fourier, fourier_periods)
}


# Recursive forecasting of the series until a given horizon
forec_recursive <- function(BASE_DIR, dataset_name, lag, model, final_lags, forecast_horizon, series_means, model_type = "pooled_regression", loss, direct, price_test_data = NULL, with_fourier, fourier_periods) {
  print(fourier_periods)
  # This will store the predictions corresponding with each horizon
  predictions <- NULL
  
  # Normalised point forecast used in quantile forecast
  if(loss == "quantile" & direct == TRUE) {
    model_50_predictions <- read.table(paste0(BASE_DIR, "/results/forecasts/", dataset_name, "_lightgbm_tweedie_lag_", lag, "_", "normalised", ".txt"), sep = ",")[,-1]
  }
  for (i in 1:forecast_horizon){
    # create fourier features
    if (with_fourier) {
      fourier_features <- NULL
      for (k in 1:4) {
        for (period in fourier_periods) {
          feature_col <- rep(sin(2.0 * pi * k * (forecast_horizon + 1 - i) / period), nrow(final_lags))
          fourier_features <- cbind(fourier_features, feature_col) 
          feature_col <- rep(cos(2.0 * pi * k * (forecast_horizon + 1 - i) / period), nrow(final_lags))
          fourier_features <- cbind(fourier_features, feature_col)
        }
      }
      final_lags <- cbind(final_lags, fourier_features)
      colnames(final_lags) <- c(paste("Lag", 1:lag, sep=""), paste0("fourier_features_", 1:(4*2*length(fourier_periods))))
    }
    # Get predictions for the current horizon
    if(model_type == "pooled_regression")
      new_predictions <- predict.glm(object = model, newdata = as.data.frame(final_lags))
    # "If you specify "response", the predictions are on the scale of the response, the inverse link function of the "link" predictions."
    else if (model_type == "glm_poisson" | model_type == "glm_negbin")
      new_predictions <- predict.glm(object = model, newdata = as.data.frame(final_lags), type = "response")
    else if (model_type == "gam")
      new_predictions <- predict.gam(object = model, newdata = as.data.frame(final_lags), type = "response")
    else if (model_type == "glmnet")
      new_predictions <- predict.glmnet.f(model, cbind(y = rep(0, nrow(final_lags)), final_lags))
    else if(model_type == "lightgbm")
      new_predictions <- predict(model, Matrix(as.matrix(final_lags), sparse = TRUE))
    
    # Adding the current forecasts to the final predictions matrix
    predictions <- cbind(predictions, new_predictions)
    
    # Updating the test set for the next horizon
    if(i < forecast_horizon){
      if (with_fourier) {
        final_lags <- final_lags[1:lag]
      }
      final_lags <- final_lags[-lag]
      
      if(loss == "quantile")
        final_lags <- cbind(model_50_predictions[,i], final_lags)
      else 
        final_lags <- cbind(new_predictions, final_lags)
      
      colnames(final_lags)[1:lag] <- paste("Lag", 1:lag, sep="")
      
      if(!is.null(price_test_data)){  
        final_lags[,(lag+1)] <- price_test_data[,(i+1)]
        colnames(final_lags)[(lag + 1)] <- "price"
      }
      
      final_lags <- as.data.frame(final_lags)
    }
  }

  # List of renormalised and normalised predictions
  predictions_renorm <- predictions * as.vector(series_means)
  list(predictions, predictions_renorm)

}
