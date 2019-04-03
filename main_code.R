## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, include = FALSE, eval = FALSE)
require(tidyverse)
require(ggpubr)
require(knitr)
require(reshape)

## ---- include = FALSE----------------------------------------------------
# To read .txt files:
read_digits <- function(directory) {
  read.table(directory, header = FALSE)
}

train <- read_digits("data/train.txt")
test <- read_digits("data/test.txt")

str(train)
dim(train)

## ------------------------------------------------------------------------
# Display the averages for all digits 0-9 of the ~7000 observations

# Convert train from a data frame to a matrix
m <- as.matrix(train)

# The image() function displays a grid of gray-scale rectangles with colors corresponding to the matrix values

# Compute the average of all pixels (by digit) into a new df
train_average <- train %>%
  group_by(V1) %>%
  summarize_all(funs(mean))

m_average <- as.matrix(train_average)

# Split the matrix into a list of the ten digits, and also reverse the values and transpose the matrix to display in the right direction
average_images <- lapply(split(m_average, row(m_average)), function(x) t(apply(matrix(x[-1], 16, 16, byrow=T), 2, rev)))

# combine all elements of average_images into a final image matrix of all the digits together
img <- do.call(rbind, 
               lapply(split(average_images, 1:5), 
                      function(x) do.call(cbind, x)))

## Plot the averages of the ten digits
image(img, col = grey(seq(1,0,length = 100)))


## --------------------------------------------------------
# total variance across all digit classes for every pixel
total_var <- data.frame(sapply(train[,2:257], var))

total_var$pixel <- c(1:256)

names(total_var) <- c("variance", "pixel")

## ------------------------------------------------------------------------
# Compute variance for every pixel by digit class
train_variance <- train %>%
  group_by(V1) %>%
  summarize_all(funs(var))

m_variance <- as.matrix(train_variance)

# Split the matrix into a list of the ten digits, and also reverse the values and transpose the matrix to display in the right direction
var_images <- lapply(split(m_variance, row(m_variance)), function(x) t(apply(matrix(x[-1], 16, 16, byrow=T), 2, rev)))

# combine all elements of var_images into a final image matrix 
img_var <- do.call(rbind, 
               lapply(split(var_images, 1:5), 
                      function(x) do.call(cbind, x)))

## Plot the total variances of all pixels of the ten digits
image(img_var, col = grey(seq(1,0,length = 100)))

## --------------------------------------------------------
# My predict_knn() function
# -----------------------------------------
predict_knn <- function(predictor, training, dist, k) {
  
  prediction = c() # initialize an empty vector to store the prediction results in
  
  # Calculate the distance using either Euclidean or Manhattan 
  get_dist = rbind(predictor, training)
  if (dist == "Euclidean") {
    Di = dist(get_dist, method = "euclidean")
    Di_m = as.matrix(Di) # Di_m is the distance matrix
  } else if (dist == "Manhattan") {
    Di = dist(get_dist, method = "manhattan")
    Di_m = as.matrix(Di) # Di_m is the distance matrix
  } else {
    message("Invalid distance metric")
  }
  
  # K Nearest neighbors
  Di_m_sort = apply(Di_m, 2, order)[1:k+1,] # sort to find the k smallest distances
  
  # Loop to iterate over every observation of the prediction set
  for (i in 1:nrow(predictor)) {
    
    # Prediction: Get the indices of the k smallest distances for the ith row
    neighbors = Di_m_sort[,i]
    # Count how many votes per class label
    votes = table(get_dist[neighbors,1])
    # Predicted class label:
    mode = names(votes)[votes == max(votes)] # find the mode of the neighbors
    if (length(mode) > 1) {
      final_mode = sample(mode, 1)
    } else {
      final_mode = mode
    }
    
    # Append the result to the prediction vector
    prediction[length(prediction) + 1] = final_mode
  }
  # return the vector of final predictions
  return(as.numeric(prediction))
}

#--------------------- 

# Testing the function on five digits

test_digit <- train[1:5,] # test on digit number six
train_digit <- train[1:500,] # 100 training rows
results <- predict_knn(predictor = test_digit, training = train_digit, dist = "Euclidean", 3)

# How many did it get right or wrong (FALSE = # incorrect)
table(results == train[1:5,1])

# Compare with actual digits
results
test_digit[,1]

## --------------------------------------------------------
# Steps for Cross Validation: 
# --------------
# 1. Shuffle the observations
shuffled_train <- train[sample(nrow(train)), ]

# 2. Split the obs into 10 folds
split_by <- rep(seq(1:10), length.out = nrow(shuffled_train))
indices <- c(1:nrow(shuffled_train))
folds <- split(indices, split_by) # folds gives the indices of the splitted data

# 3. Computing the distance matrices BEFORE running loop:

# Computing the distance matrices before running cv function:

# calculate the Euclidean distance matrix
dist_e <- dist(train, method = "euclidean") # euclidean dist matrix
dist_m_e <- as.matrix(dist_e) # convert to matrix
dim(dist_m_e)
dist_m_e_sort <- apply(dist_m_e, 2, function(x)
  order(x))

# calculate the Manhattan distance matrix
dist_m <- dist(train, method = "manhattan")
dist_m_m <- as.matrix(dist_m)
dist_m_m_sort <- apply(dist_m_m, 2, function(x)
  order(x))

# dist_m_e_sort and dist_m_m_sort are the sorted indices of the Euclidean and Manhattan distance 

dim(dist_m_e_sort) # check: this is a 7291x7291 matrix of indices

cv_error_knn <- function(k) {                  # accepts k as a parameter
  mean(sapply(folds, function(fold) {          # take the mean of the 10 error rates to get the overall cv error rate
  d = dist_m_e_sort[-fold, fold]
  train_labels = labels(-fold)
  # knn function
  knn2 <- function(predictor, training, k) {
    prediction = c() # initialize an empty vector to store the prediction results in
    # K Nearest neighbors
    Di_m_sort = d[1:k+1,] # sort to find the k smallest distances
    # Loop to iterate over every observation of the prediction set
    for (i in 1:nrow(predictor)) {
      # Prediction:
      # Get the indices of the k smallest distances for the ith row
      neighbors = Di_m_sort[,i]
      # Count how many votes per class label
      votes = table(train[neighbors,1])
      # Predicted class label:
      mode = names(votes)[votes == max(votes)] # find the mode of the neighbors
      if (length(mode) > 1) {
        final_mode = sample(mode, 1)
      } else {
        final_mode = mode
      }
      prediction[length(prediction) + 1] = final_mode
    }
    # vector of final predictions
    return(as.numeric(prediction))
  }
  
  final_pred <- knn2(train[fold,], train, k) # store the predictions 
  
  error_rate <- sum(final_pred != train[fold,1])/length(final_pred) # error rate is the sum of mismatches divided by the length 
  
})) 
}

cv_error_knn(3) # test the function

## --------------------------------------------------------
# Iterate over values k=1..15 and two different distance metrics

# create new cv error function for using Manhattan distance
cv_error_knn_man <- function(k) {                  # accepts k as a parameter
  mean(sapply(folds, function(fold) {          # take the mean of the 10 error rates to get the overall cv error rate
    d = dist_m_m_sort[-fold, fold]
    train_labels = labels(-fold)
    # knn function
    knn2 <- function(predictor, training, k) {
      prediction = c() # initialize an empty vector to store the prediction results in
      # K Nearest neighbors
      Di_m_sort = d[1:k+1,] # sort to find the k smallest distances
      # Loop to iterate over every observation of the prediction set
      for (i in 1:nrow(predictor)) {
        # Prediction:
        # Get the indices of the k smallest distances for the ith row
        neighbors = Di_m_sort[,i]
        # Count how many votes per class label
        votes = table(train[neighbors,1])
        # Predicted class label:
        mode = names(votes)[votes == max(votes)] # find the mode of the neighbors
        if (length(mode) > 1) {
          final_mode = sample(mode, 1)
        } else {
          final_mode = mode
        }
        prediction[length(prediction) + 1] = final_mode
      }
      # vector of final predictions
      return(as.numeric(prediction))
    }
    
    final_pred <- knn2(train[fold,], train, k) # store the predictions 
    error_rate <- sum(final_pred != train[fold,1])/length(final_pred) # error rate is the sum of mismatches divided by the length 
    
  })) 
}

cv_error_knn_man(3) #test function

# ------------------------
# compute the error rates for k = 1..15 for Euclidean and Manhattan metrics

k_error_rates_e <- sapply(2:15, cv_error_knn) # k error rates for Euclidean
k_error_rates_e <- c(NA, k_error_rates_e)

k_error_rates_m <- sapply(2:15, cv_error_knn_man) # k error rates for Manhattan
k_error_rates_m <- c(NA, k_error_rates_m)

x_axis <- seq(1,15)
df <- data.frame(x_axis, k_error_rates_e, k_error_rates_m)
df2 <- melt(df, id = c("x_axis"))
names(df2) <- c("x_axis", "type", "error_rate")
df2

error_plot <- ggplot(df2, aes(x = x_axis, y = error_rate)) + 
  geom_point(size = 3, aes(color = type)) +
  labs(title="Cross Validation Error Rates for K", x = "k", y = "Error Rates") + 
  scale_color_discrete(name = "Distance Metric", labels = c("Euclidean", "Manhattan"), guide = guide_legend(reverse=TRUE))


## ---- echo = TRUE--------------------------------------------------------
error_plot

## ------------------------------------------------------------------------
# Test set error rates
# -------------------
# test for Euclidean
test_error_e <- function(k) {                  
    # knn function
  test_answer <- predict_knn(test[1:100,], train[1:100,], "Euclidean", k) # store predictions 
  test_error_rate <- sum(test_answer != test[,1])/length(test_answer) # error rate is the sum of mismatches divided by the length 
}

k_test_error_rates <- sapply(2:15, test_error_e) # test error rates for Euclidean
k_test_error_rates <- c(NA, k_test_error_rates)
# ----------------------------------
# test for Manhattan
test_error_m <- function(k) {                  
  # knn function
  test_answer <- predict_knn(test[1:100,], train[1:100,], "Manhattan", k) # store predictions 
  test_error_rate <- sum(test_answer != test[,1])/length(test_answer) # error rate is the sum of mismatches divided by the length 
}

k_test_error_rates_m <- sapply(2:15, test_error_m) # test error rates for Manhattan
k_test_error_rates_m <- c(NA, k_test_error_rates_m)

x_axis <- seq(1,15)
test_df <- data.frame(x_axis, k_test_error_rates, k_test_error_rates_m)
test_df2 <- melt(df, id = c("x_axis"))
names(test_df2) <- c("x_axis", "type", "error_rate")
test_df2

test_error_plot <- ggplot(test_df2, aes(x = x_axis, y = error_rate)) + 
  geom_point(size = 3, aes(color = type)) +
  labs(title="Test Error Rates for K", x = "k", y = "Error Rates") + 
  scale_color_discrete(name = "Distance Metric", labels = c("Euclidean", "Manhattan"), guide = guide_legend(reverse=TRUE))

test_error_plot

## ------------------------------------------------------------------------

# what kind of digits does the best model get wrong?
# Best model for CV is k=3 and distance=Euclidean

cv_error_knn_errors <- function(k) {                  # accepts k as a parameter
  sapply(folds, function(fold) {          
    d = dist_m_e_sort[-fold, fold]
    train_labels = labels(-fold)
    # knn function
    knn2 <- function(predictor, training, k) {
      prediction = c() # initialize an empty vector to store the prediction results in
      # K Nearest neighbors
      Di_m_sort = d[1:k+1,] # sort to find the k smallest distances
      # Loop to iterate over every observation of the prediction set
      for (i in 1:nrow(predictor)) {
        # Prediction:
        # Get the indices of the k smallest distances for the ith row
        neighbors = Di_m_sort[,i]
        # Count how many votes per class label
        votes = table(train[neighbors,1])
        # Predicted class label:
        mode = names(votes)[votes == max(votes)] # find the mode of the neighbors
        if (length(mode) > 1) {
          final_mode = sample(mode, 1)
        } else {
          final_mode = mode
        }
        prediction[length(prediction) + 1] = final_mode
      }
      # vector of final predictions
      return(as.numeric(prediction))
    }
    
    final_pred <- knn2(train[fold,], train, k) # store the predictions 
    errors <- which(final_pred != train[fold,1]) # errors are the mismatches 
    train[errors,1]
    
  })
}

# the digits that are most often gotten wrong
often_wrong <- cv_error_knn_errors(3) 
often_wrong[[1]]

