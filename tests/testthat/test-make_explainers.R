test_that("test-make_explainers", {
  data("adult")
  adult<-adult[1:800,]
  sensitive = adult$sex
  sensitive = as.integer(sensitive)

  data <- adult[ , c(2,3,4,5,6,7,8,9,12,13,14,15)]
  data$workclass <- as.integer(data$workclass)
  data$education  <- as.integer(data$education)
  data$marital_status <- as.integer(data$marital_status)
  data$occupation <- as.integer(data$occupation)
  data$relationship <- as.integer(data$relationship)
  data$native_country <- as.integer(data$native_country)

  data_matrix=matrix(unlist(data),ncol=12)
  data_scaled=scale(data_matrix,center=TRUE,scale=TRUE)

  target=adult$salary
  target= as.integer(target)
  set.seed(7)
  train_indices <- sample(1:nrow(adult), 0.7*nrow(adult))
  adult_test<-adult[setdiff(1:nrow(adult), train_indices),]
  data_scaled_test<-data_scaled[setdiff(1:nrow(adult), train_indices),]

  train_x<-data_scaled[train_indices,]
  test_x<-data_scaled[setdiff(1:nrow(data_scaled), train_indices), ]

  train_y<-target[train_indices]
  test_y<-target[setdiff(1:length(target), train_indices)]

  sensitive_train<-sensitive[train_indices]
  sensitive_test<-sensitive[setdiff(1:length(sensitive), train_indices)]

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(train_x, train_y, test_x, test_y, 5, dev)

  clf_model <- create_model(train_x,train_y, c(8,8,8), dimensions = 2)
  clf_model$to(device = dev)

  sil <- pretrain_net(10, clf_model, dsl, model_type = 1, 0.001,
                        sensitive_test, dev,verbose = FALSE, monitor = FALSE)

  clf_model2 <- create_model(train_x,train_y, c(8,8,8), dimensions = 2)
  clf_model2$to(device = dev)

  sil <- pretrain_net(5, clf_model2, dsl, model_type = 1, 0.001,
                        sensitive_test, dev,verbose = FALSE, monitor = FALSE)

  clf_model3 <- create_model(train_x,train_y, c(16,16,16), dimensions = 2)
  clf_model3$to(device = dev)

  sil <- pretrain_net(7, clf_model3, dsl, model_type = 1, 0.001,
                        sensitive_test, dev,verbose = FALSE, monitor = FALSE)

  single <- Single_explainer(test_y,clf_model,"classifier",adult_test,data_scaled_test,protected=
                     adult_test$sex,privileged = "Male",batch_size = 5,dev=dev, verbose = FALSE)

  dual <- Dual_explainer(test_y,clf_model,clf_model2,"classifier","classifier2",adult_test,
                           data_scaled_test,protected=adult_test$sex,privileged = "Male",
                           batch_size = 5,dev=dev, verbose = FALSE )

  triple <- Triple_explainer(test_y,clf_model,clf_model2,clf_model3,"classifier","classifier2",
                             "classifier3",adult_test,data_scaled_test,protected=adult_test$sex,
                             privileged = "Male",batch_size = 5,dev=dev, verbose = FALSE )

  group_metric_matrix <- matrix(0, nrow = 1 , ncol = 12)
  colnames(group_metric_matrix) <- c("TPR","TNR","PPV","NPV","FNR","FPR","FDR","FOR","TS","STP","ACC","F1")
  group_metric_matrix[1,] <- c(0.607989 ,0.001898,0.810930 ,0.148088 ,0.23469 ,0.032335,
                               0.980829 ,0.766759,0.828949 ,0.711839 ,0.104541 ,0.679354)

  group_metric_matrix2 <- matrix(0, nrow = 2 , ncol = 12)
  colnames(group_metric_matrix2) <- c("TPR","TNR","PPV","NPV","FNR","FPR","FDR","FOR","TS","STP","ACC","F1")
  group_metric_matrix2[1,] <- c(0.607989 ,0.001898,0.810930 ,0.148088 ,0.23469 ,0.032335,
                               0.980829 ,0.766759,0.828949 ,0.711839 ,0.104541 ,0.679354)
  group_metric_matrix2[2,] <- c(NaN,0,NA,0.236655 ,0,NaN,NA,0.914780,NaN,NaN,0.236655,NA)

  group_metric_matrix3 <- matrix(0, nrow = 3 , ncol = 12)
  colnames(group_metric_matrix3) <- c("TPR","TNR","PPV","NPV","FNR","FPR","FDR","FOR","TS","STP","ACC","F1")
  group_metric_matrix3[1,] <- c(0.607989 ,0.001898,0.810930 ,0.148088 ,0.23469 ,0.032335,
                                0.980829 ,0.766759,0.828949 ,0.711839 ,0.104541 ,0.679354)
  group_metric_matrix3[2,] <- c(NaN,0,NA,0.236655 ,0,NaN,NA,0.914780,NaN,NaN,0.236655,NA)
  group_metric_matrix3[3,] <- c(0.607989 ,0.020946,0.897942 ,0.143746 ,0.23469 ,0.437800,
                                1.299283 ,0.752053,0.865990 ,0.624828 ,0.088280 ,0.707133)

  expect_true(single$epsilon==0.8)
  expect_true(single$privileged=="Male")
  expect_true(sum(single$protected==adult_test$sex)==length(adult_test$sex))
  expect_true(single$label=="classifier")
  expect_true( max(group_metric_matrix-round(single$parity_loss_metric_data,6))==
                 min(group_metric_matrix-round(single$parity_loss_metric_data,6)) )


  expect_true(dual$epsilon==0.8)
  expect_true(dual$privileged=="Male")
  expect_true(sum(dual$protected==adult_test$sex)==length(adult_test$sex))
  expect_true(sum(dual$label==c("classifier","classifier2"))==2)

  helper <- (group_metric_matrix2-round(dual$parity_loss_metric_data,6))==
    (group_metric_matrix2-round(dual$parity_loss_metric_data,6))

  for(i in 1:2){
    for(j in 1:12){
      if(is.na(helper[i,j])){
        helper[i,j]<-TRUE
      }
    }
  }

  expect_true( sum(helper) == 24)

  expect_true(triple$epsilon==0.8)
  expect_true(triple$privileged=="Male")
  expect_true(sum(triple$protected==adult_test$sex)==length(adult_test$sex))
  expect_true(sum(triple$label==c("classifier","classifier2","classifier3"))==3)

  helper <- (group_metric_matrix3-round(triple$parity_loss_metric_data,6))==
    (group_metric_matrix3-round(triple$parity_loss_metric_data,6))

  for(i in 1:3){
    for(j in 1:12){
      if(is.na(helper[i,j])){
        helper[i,j]<-TRUE
      }
    }
  }

  expect_true( sum(helper) == 36)


})
