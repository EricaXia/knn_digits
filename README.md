# Classification of Handwritten Digits using K-Nearest-Neighbors in R

Data: A collection of handwritten numerical digits 

Model: K-Nearest Neighbors Classifier

The k-nearest neighbors algorithm classifies an observation based on the class labels of the k nearest observations in the training set (the “neighbors”). The classification power of k-nn depends on the choice of k and on the metric used to measure distance. Common
distance metrics include include Euclidean, Manhattan, and Minkowski distance.

## Graphically Displaying the Average Digit

![alt text](https://github.com/EricaXia/knn_digits/blob/master/images/knn1.PNG "Average Digit")

The above image shows what each digit looks like on average.

## Which pixels in the image are best for classification?

![alt text](https://github.com/EricaXia/knn_digits/blob/master/images/knn2.PNG "Average Digit")

This combined variances image plots the variance for every pixel for each digit. The light areas correspond to the pixels with the lowest variance. The dark areas indicate those pixels have high variance. Therefore, the lighter pixels shown in the image are the most
useful for classification. Pixels in the dark ares with higher variance are least likely to be useful for classification.

## How the KNN Function Works

## 10-fold Cross Validation to Estimate the Error Rate for KNN

In order to make cross validation run efficiently, I first calculated the distance matricies (for Euclidean and Manhattan distance metrics) beforehand, then ordered the matricies from least to greatest and obtained the index labels. The data was shuffled
randomly and split into ten folds by assigning each observation a number from 1 to 10. The folds were stored in a list of indicies.
My cv_error_knn function applies the same process to each of the ten folds: the distance matrix is subsetted using the current fold, before being passed to the knn function. My knn function had to be altered slightly so it would accept a distance matrix as a
parameter, rather than a distance metric. Each iteration of the loop resulted in a vector of predictions which is compared to the real values to estimate an error rate. Finally, I took the average of the ten error rates to get my overall error rate. For k = 3 neighbors, my
estimated cross validation error is about 0.01495, or 1.5%. (The error will differ slightly each time the function is run, but it stays around the same range.)
