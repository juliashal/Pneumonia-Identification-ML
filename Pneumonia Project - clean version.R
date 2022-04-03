library(tidyverse)
library(imager)
library(opencv)
library(caret)
library(dplyr)
library(matrixStats)
library(randomForest)
library(rpart)

# The initial data source is https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# However, it's quite complicated to download the archive directly from the website.
# KEY - get data from Google drive

install.packages("googledrive")
library("googledrive")


# links to 3 ZIP datasets (train, test, val) (Google Drive)
# These 3 datasets were split by the author Paul Mooney

train_url <- "https://drive.google.com/file/d/1UEdxXC2JuQ9ZK9ZcuH1Txa14xn0aFpc9/download"
test_url <- "https://drive.google.com/file/d/1UF6JKw7O4qSwxOrIp-k3YXp_a-15EDGr/download"
val_url <- "https://drive.google.com/file/d/1UEDMUIwL_8RIOK4fMPP7lTFv8p2VM_sH/download"

val_zip <- "val.zip"
test_zip <- "test.zip"
train_zip <- "train.zip"

val_folder <- "val"
test_folder <- "test"
train_folder <- "train"

#####################
# Data Exploration
#####################



#####################
# Part 1 : Load data
#####################

# Let's have a look at the examples of chest xrays
drive_download(val_url)
out <- unzip(val_zip)
list_subfolders <- list.dirs(path = val_folder, full.names = TRUE, recursive = FALSE)
filenames <- lapply(list_subfolders, function(x) list.files(path = x, pattern = "*.jpeg", full.names = TRUE))
filenames <- do.call(c, filenames)
file.remove(val_zip) # delete zip

Normal <- load.image(filenames[1])
Pneumonia <- load.image(filenames[9])
unlink(val_folder, recursive = TRUE) # delete temp photos

layout(t(1:2))
plot(Normal, main = "Normal Xray example")
plot(Pneumonia, main = "Pneumonia Xray example")

# We can see that the normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image,
# meanwhile viral pneumonia (right) manifests with a  ‘‘foggier’’ pattern in both lungs

# Dimensions of photos
Normal
Pneumonia

# As dimensions are different we need to resize all photos and load to R
# Note: during investigation it was found out that some photos have 3 color channels (RGB, not black&white), as a result it was decided to convert all photos to grayscale as well
rm(Normal, Pneumonia, filenames, out, list_subfolders)


# Function to unzip, load and adjust photos
process_folder <- function(url, zip, folder) {

  # 1 Step - download ZIP
  drive_download(url)

  # 2 Step - extract files from archive (to working directory)
  out <- unzip(zip)

  # 3 Step - go to the folder and load all photos to R
  list_subfolders <- list.dirs(path = folder, full.names = TRUE, recursive = FALSE)
  filenames <- lapply(list_subfolders, function(x) list.files(path = x, pattern = "*.jpeg", full.names = TRUE))
  filenames <- do.call(c, filenames)

  # Function to load all photos to R, turn t into gray scale and resize
  temp <- lapply(filenames, function(x) {
    im <- load.image(x)
    im <- grayscale(im)
    im <- resize(im, 100, 100)
    temp <- c(as.data.frame(im)$value * 255.0, Y = basename(dirname(x)))
  })

  temp <- as.data.frame(do.call("rbind", temp))
  unlink(folder, recursive = TRUE) # delete temp photos
  file.remove(zip) # delete zip
  temp
}

val <- process_folder(val_url, val_zip, val_folder)
test <- process_folder(test_url, test_zip, test_folder)
train <- process_folder(train_url, train_zip, train_folder)


# Remove all unnecessary variables
rm(test_folder, test_url, test_zip, train_folder, train_url, train_zip, val_folder, val_url, val_zip)

# Example of resized and adjusted photos
rafalib::mypar()
normal <- 1
pneumonia <- 9

m1 <- matrix(val[normal, 1:10000], 100, 100)
mode(m1) <- "numeric"
m2 <- matrix(val[pneumonia, 1:10000], 100, 100)
mode(m2) <- "numeric"

layout(t(1:2))
image(m1, main = "Normal Xray example") # to show darkest pixels
image(m2, main = "Pneumonia Xray example")

rm(m1, m2, normal, pneumonia)


#####################
# Part 2 : Dataset exploration
#####################
dim(train)
dim(test)
dim(val)

# Data split
data.frame(train = nrow(train), test = nrow(test), val = nrow(val)) %>%
  gather(Dataset, Obs) %>%
  mutate(percentage = round(Obs / sum(Obs), 3))
# Not aligned with a best practice rule 80-10-10 (80% train, 10% test, and 10% validation)

# Check share of each category ("Normal", "Pneumonia")
train_stats <- train %>%
  group_by(Y) %>%
  summarize(n = n()) %>%
  mutate(type = "train", percent = n / sum(n))
test_stats <- test %>%
  group_by(Y) %>%
  summarize(n = n()) %>%
  mutate(type = "test", percent = n / sum(n))
val_stats <- val %>%
  group_by(Y) %>%
  summarize(n = n()) %>%
  mutate(type = "val", percent = n / sum(n))

all_db_stats <- rbind(train_stats, test_stats, val_stats)

all_db_stats %>% ggplot(aes(fill = Y, y = percent, x = type)) +
  geom_bar(position = "fill", stat = "identity") +
  geom_text(aes(label = paste0(sprintf("%1.1f", percent * 100), "%")), position = position_stack(vjust = 0.5), colour = "white", size = 5)


# Graph shows that shares of Y differs between sets, especially % of Pneumonia photos is higher in the Train set.
# This fact may cause models over train and bias to detect pneumonia cases.
# We need to adjust split not only to 80-10-10 rules, but also that each set represents population accurately.

rm(train_stats, test_stats, val_stats, full_stats)

#####################
# Part 3 : Dataset re-split
#####################

full_db <- rbind(train, test, val) # combine all photos

set.seed(2007)
test_index <- createDataPartition(full_db$Y, times = 1, p = 0.2, list = FALSE)

# Split db into train and test (80/20)

train_set_new <- full_db[-test_index, ]
test_set_temp <- full_db[test_index, ]

# Split test set into test and validation datasets (10/10)

val_index <- createDataPartition(test_set_temp$Y, times = 1, p = 0.5, list = FALSE)
test_set_new <- test_set_temp[-val_index, ]
val_set_new <- test_set_temp[val_index, ]

train_set_new[, 1:10000] <- sapply(train_set_new[, 1:10000], as.numeric)
test_set_new[, 1:10000] <- sapply(test_set_new[, 1:10000], as.numeric)
val_set_new[, 1:10000] <- sapply(val_set_new[, 1:10000], as.numeric)

# Let's check if data in each set has the same share of Normal and Pneumonia photos

train_stats_new <- train_set_new %>%
  group_by(Y) %>%
  summarize(n = n()) %>%
  mutate(type = "train", percent = n / sum(n))
test_stats_new <- test_set_new %>%
  group_by(Y) %>%
  summarize(n = n()) %>%
  mutate(type = "test", percent = n / sum(n))
val_stats_new <- val_set_new %>%
  group_by(Y) %>%
  summarize(n = n()) %>%
  mutate(type = "val", percent = n / sum(n))

full_stats_new <- rbind(train_stats_new, test_stats_new, val_stats_new)

full_stats_new %>% ggplot(aes(fill = Y, y = percent, x = type)) +
  geom_bar(position = "fill", stat = "identity") +
  geom_text(aes(label = paste0(sprintf("%1.1f", percent * 100), "%")), position = position_stack(vjust = 0.5), colour = "white", size = 5)

# Now it's much better

rm(train_stats_new, test_stats_new, val_stats_new, full_stats_new)
rm(test_set_temp, test_index, val_index)

# Split x and y
train_x <- train_set_new[, 1:10000]
train_y <- train_set_new[, 10001]

test_x <- test_set_new[, 1:10000]
test_y <- test_set_new[, 10001]

val_x <- val_set_new[, 1:10000]
val_y <- val_set_new[, 10001]


#####################
# Data Preprocessing
#####################


#####################
# Part 1 : Remove predictors that are not useful
#####################

library(matrixStats)

nzv <- nearZeroVar(train_x) # recommends features to be removed due to near zero variance
image(matrix(1:10000 %in% nzv, 100, 100))

# We can see that pixels on edges of xray doesn't bring useful information.
# We will help model to cut borders more as we see that around 10 pixels from left and right will not even touch chest area.

seq_new <- seq(0, 9900, 100)
pixels_to_cut <- c(0:9, 90:99)
new_nzv <- c()
for (i in seq_new) {
  for (j in pixels_to_cut) {
    new_nzv <- c(new_nzv, i + j)
  }
}
new_nzv <- c(new_nzv, 10000)

# Revised pixels crop
layout(t(1:2))
image(matrix(1:10000 %in% nzv, 100, 100))
image(matrix(1:10000 %in% new_nzv, 100, 100))

# Let's also cut first bottom 2000 pixels as they usually contain the pelvic bone and are not related to the topic of the research.
#+ top 2000 pixels covering neck and head

new_nzv <- c(new_nzv, c(1:2000), c(8000:10000))
new_nzv <- unique(new_nzv)

# Revised pixels crop - Final
layout(t(1:2))
image(matrix(1:10000 %in% nzv, 100, 100), main = "First Crop")
image(matrix(1:10000 %in% new_nzv, 100, 100), main = "Revised Crop")

col_index_1_cut <- setdiff(1:ncol(train_x), new_nzv) # indexes left after 1st cut

# The dimension of pixels left would be 80x60
# Examples
normal <- 585
pneumonia <- 4290

m1 <- matrix(train_x[normal, !(1:10000 %in% new_nzv)], 80, 60)
mode(m1) <- "numeric"
m2 <- matrix(train_x[pneumonia, !(1:10000 %in% new_nzv)], 80, 60)
mode(m2) <- "numeric"

layout(t(1:2))
image(m1, main = "Normal")
image(m2, main = "Pneumonia")

rm(m1, m2,normal,pneumonia)
#####################
# Part 2 : Normalization
#####################

# Before performing any model, we need to normalize our factors by deducting AVG and dividing by SD.
# It's required so as algorithm would not be dominated by the variables that use a larger scale, adversely affecting model performance

colMeans <- colMeans(as.matrix(train_x))
colSds <- colSds(as.matrix(train_x))

layout(t(1:1))
image(matrix(colSds[col_index_1_cut], 80, 60), main = "Std deviation of pixels")
# From matrix of St deviations we can see that lowest deviation is the area of heart and spine,
# more deviation around ribs area,
# and highest on the borders there photos usually differ from each other

# Need to exclude Indexes with lowest Std
layout(t(1:2))
hist(colSds[col_index_1_cut], main = "Histogram of Std Deviation", xlab = "Std dev")
image(matrix(colSds[col_index_1_cut] < 30, 80, 60))

# We can see that the 1st hill in the histogram of Std deviation is lowest deviation around heart and spine
# it's another 1202 variables to exclude from the model (sds<30)
index_to_exclude <- which(colSds[col_index_1_cut] < 30)
col_index <- col_index_1_cut[-index_to_exclude]
# Now we have 3598 variables out of 10000

# Ready to normalize
train_x_scaled <- sweep(sweep(train_x, 2, colMeans), 2, colSds, FUN = "/")
test_x_scaled <- sweep(sweep(test_x, 2, colMeans), 2, colSds, FUN = "/")
val_x_scaled <- sweep(sweep(val_x, 2, colMeans), 2, colSds, FUN = "/")



#####################
# Model testing
#####################


#####################
# Part 1 : K-Means
#####################


# First, let's test different # of clusters for K-means and how magnitude of WSS (distance between data points the centroids of clusters) would change with # clusters
set.seed(3, sample.kind = "Rounding")
wss <- NULL
for (i in c(2, 4, 6, 8, 10, 12, 14)) {
  k <- kmeans(train_x_scaled[, col_index], centers = i, nstart = 25)
  wss <- c(wss, k$tot.withinss) # WSS as measure of compactness
}
plot(c(2, 4, 6, 8, 10, 12, 14), wss, type = "o")
# Approx Time = 40 mins
# Based on Elbow rule, k=6 would be an optimal # clusters


k <- kmeans(train_x_scaled[, col_index], centers = 6, nstart = 25) # only 2 centers since we have only 2 category option (Normal, Pneumonia)

# Function to predict cluster
predict_kmeans <- function(x, k) {
  centers <- k$centers # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i) {
    apply(centers, 1, function(y) dist(rbind(x[i, ], y)))
  })
  max.col(-t(distances)) # select cluster with min distance to center
}

y_hat_k_means <- predict_kmeans(test_x_scaled[, col_index], k)
y_hat_k_means <- ifelse(y_hat_k_means == 1, "NORMAL", "PNEUMONIA")

cm_k_means <- confusionMatrix(as.factor(y_hat_k_means), as.factor(test_y))
cm_k_means
F1_K_means <- F_meas(as.factor(y_hat_k_means), as.factor(test_y))
F1_K_means
# approx time = 10 mins
# very low F1 score



#####################
# Part 2 : Logistic Regression
#####################

library(glmnet)
set.seed(1, sample.kind = "Rounding")

train_glmnet <- cv.glmnet(x = as.matrix(train_x_scaled[, col_index]), y = train_y, family = "binomial", type.measure = "class")
plot(train_glmnet)
train_glmnet$lambda.1se # we choose lambda with lowest misclassification error

y_hat_glmnet <- predict(train_glmnet, as.matrix(test_x_scaled[, col_index]), s = "lambda.1se", type = "class")
y_hat_glmnet <- as.vector(y_hat_glmnet)
cm_glmnet <- confusionMatrix(data = as.factor(y_hat_glmnet), reference = as.factor(test_y))
cm_glmnet

F1_glmnet <- F_meas(as.factor(y_hat_glmnet), as.factor(test_y))
F1_glmnet

glmnet_coef <- as.matrix(coef(train_glmnet, s = "lambda.1se"))

glmnet_coef <- glmnet_coef %>%
  data.frame() %>%
  mutate(feature = row.names(.)) %>%
  mutate(feature = str_replace(feature, "V", "")) %>%
  mutate(index = as.integer(feature))

glmnet_coef_whole_image <- left_join(data.frame(index = col_index_1_cut), glmnet_coef, by = "index")
glmnet_coef_whole_image[is.na(glmnet_coef_whole_image)] <- 0
layout(t(1:2))
image(matrix(colSds[col_index_1_cut], 80, 60), # most variable factors
  main = paste0("Most variable predictors")
)
image(matrix(glmnet_coef_whole_image$s1, 80, 60), # Highest coef importance
  main = paste0("Highest Coef importance in Logistic regression")
) # the lighter the less probability it will have on Pneumonia


# Appr time 5 mins

#####################
# Part 3 : Classification and Decision Tree
#####################

set.seed(1, sample.kind = "Rounding")

control <- trainControl(method = "cv", number = 10, p = .9) # Use cross validation to make speed quicker

train_part <- train(
  x = train_x_scaled[, col_index], y = train_y,
  method = "rpart",
  tuneLength = 15,
  trControl = control
) # will test 15 different cp (complexity parameters) to avoid over-fitting

train_part
plot(train_part) # shows complexity parameters and accuracy of the model
# Largest Accuracy us with cp = 0.005134281.

y_hat_rpart <- predict(train_part, test_x_scaled[, col_index])

cm_rpart <- confusionMatrix(data = y_hat_rpart, reference = factor(test_y))
cm_rpart
F1_K_rpart <- F_meas(as.factor(y_hat_rpart), as.factor(test_y))
F1_K_rpart

# Appr time 7 mins

#####################
# Part 4 : KNN
#####################

set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9) # Use cross validation to make speed quicker

train_knn <- train(train_x_scaled[, col_index], train_y,
  method = "knn",
  tuneGrid = data.frame(k = seq(1, 12, 2)),
  trControl = control
)
ggplot(train_knn)
train_knn$bestTune
train_knn$finalModel

# Check the model on the test set

y_hat_knn <- predict(train_knn, test_x_scaled[, col_index], type = "raw")
cm_knn <- confusionMatrix(data = y_hat_knn, reference = factor(test_y))
cm_knn

# Appr time 84 mins

F1_knn <- F_meas(y_hat_knn, factor(test_y))
F1_knn


# Visualize where we made mistakes
fit_knn <- knn3(train_x_scaled[, col_index], factor(train_y), k = 5)
p_max <- predict(fit_knn, test_x_scaled[, col_index]) # Probability of prediction
p_max <- apply(p_max, 1, max)
ind <- which(y_hat_knn != factor(test_y)) # indexes where model failed
ind <- ind[order(p_max[ind], decreasing = TRUE)] # examples with highest errors
rafalib::mypar(2, 3)
for (i in ind[1:6]) {
  image(matrix(data.matrix(test_x_scaled[i, 1:10000]), 100, 100)[, 100:1],
    main = paste0(
      "Pr(", y_hat_knn[i], ")=", round(p_max[i], 2),
      " but is a ", test_y[i]
    ),
    xaxt = "n", yaxt = "n"
  )
}


#####################
# Part 5 : Random Forest
#####################

set.seed(9, sample.kind = "Rounding")
# Tune mtry - Number of variables randomly sampled as candidates at each split.

bestmtry <- tuneRF(train_x_scaled[, col_index], as.factor(train_y), stepFactor = 1.5, improve = 0.01, ntree = 500)
print(bestmtry)

# In the results of the code we can see that the lowest OOB (Out-of-bag) error (prediction error) appears with 88 random variables for Random forest
# We'll use it as input into our final model
bestmtry <- data.frame(bestmtry)
final_mtry <- bestmtry$mtry[which.min(bestmtry$OOBError)]

fit_rf <- randomForest(train_x_scaled[, col_index], as.factor(train_y),
  mtry = final_mtry, importance = TRUE
)

# To get plot with error vs # trees created
plot(fit_rf)
# Details about the model
fit_rf

# However, there is a room for improvement here: let's try to make the estimate smoother by changing # of data points in nodes
nodesize <- seq(1, 51, 10)
rf_accuracy <- sapply(nodesize, function(ns) {
  fit_rf_temp <- randomForest(train_x_scaled[, col_index], as.factor(train_y),
    mtry = final_mtry,
    nodesize = nodesize,
    importance = TRUE
  )
  y_hat_rf_temp <- predict(fit_rf_temp, test_x_scaled[, col_index])
  F_meas(y_hat_rf_temp, as.factor(test_y))
})
qplot(nodesize, rf_accuracy)

# Appr time=2 hrs

nodesize_best <- nodesize[which.max(rf_accuracy)]
# Based on these calculations, the highest F1 index is achieved with 11 nodes in a tree.
# We'll use it in our final model for Random Forest

fit_rf <- randomForest(train_x_scaled[, col_index], as.factor(train_y),
  mtry = final_mtry,
  nodesize = nodesize_best,
  importance = TRUE
)

# Appr time=20 mins


varImpPlot(fit_rf) # most important variables

# Export important variables indexes and Mean Decrease Gini index
feat_imp_df <- importance(fit_rf) %>%
  data.frame() %>%
  mutate(feature = row.names(.)) %>%
  mutate(feature = str_replace(feature, "V", "")) %>%
  mutate(index = as.integer(feature))

# Mean Decrease Gini - Measure of variable importance based on the Gini impurity index
# Gini impurity index calculates the amount of probability of a specific feature that is classified incorrectly when selected randomly. If all the elements are linked with a single class then it can be called pure.
# The higher MDGini is, the higher the importance of the variable in the model.
MDGini <- left_join(data.frame(index = col_index_1_cut), feat_imp_df, by = "index")
MDGini[is.na(MDGini)] <- 0

layout(t(1:2))
image(matrix(colSds[col_index_1_cut], 80, 60),
  main = paste0("Most variable predictors")
)
image(matrix(MDGini$MeanDecreaseGini, 80, 60),
  main = paste0("Highest Gini index in Random Forest")
)

# Check the model on test set
y_hat_rf <- predict(fit_rf, test_x_scaled[, col_index])
cm_rf <- confusionMatrix(y_hat_rf, as.factor(test_y))
cm_rf
F1_RF <- F_meas(y_hat_rf, as.factor(test_y))
F1_RF

# We can see that sensitivity has been improved dramatically (by +10 pp).
# But we also see that we predict less Pneumonia cases accurately (Neg pred value)


#####################
# Part 6 : GLM + Random Forest (best models, >=80% F1)
#####################

p_glmnet <- predict(train_glmnet, as.matrix(test_x_scaled[, col_index]), s = "lambda.1se", type = "response")
p_rf <- predict(fit_rf, test_x_scaled[, col_index], type = "prob")
p <- (p_rf[, 2] + p_glmnet) / 2

y_ensemble_hat <- as.factor(ifelse(p >= 0.5, 2, 1))
test_y_factor <- as.factor(ifelse(test_y == "NORMAL", 1, 2))

cm_ensemble <- confusionMatrix(y_ensemble_hat, test_y_factor)
cm_ensemble
F1_ensemble <- F_meas(y_ensemble_hat, test_y_factor)
F1_ensemble

results <- data.frame(GlmNet = round(F1_glmnet, 3), Rpart = round(F1_K_rpart, 3), KNN = round(F1_knn, 3), RF = round(F1_RF, 3), Glm_RF = round(F1_ensemble, 3))
results <- results %>%
  gather(Model, F1) %>%
  arrange(desc(F1))
results

#####################
# END: Model implementation on validation dataset
#####################
p_glmnet_final <- predict(train_glmnet, as.matrix(val_x_scaled[, col_index]), s = "lambda.1se", type = "response")
p_rf_final <- predict(fit_rf, val_x_scaled[, col_index], type = "prob")
p_final <- (p_rf_final[, 2] + p_glmnet_final) / 2

y_hat_val <- as.factor(ifelse(p_final>= 0.5, 2, 1))
val_y_factor<- as.factor(ifelse(val_y == "NORMAL", 1, 2))

cm_final <- confusionMatrix(y_hat_val, val_y_factor)
cm_final
F1_final <- F_meas(y_hat_val, val_y_factor)
F1_final