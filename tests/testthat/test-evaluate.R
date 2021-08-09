test_that("test-evaluate", { #printy irytują
  data("adult")
  adult<-adult[1:200,]
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

  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  acc1 <- eval_accuracy(clf_model,dsl$test_ds,dev)
  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  acc2 <- eval_accuracy(clf_model,dsl$test_ds,dev)
  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  acc3 <- eval_accuracy(clf_model,dsl$test_ds,dev)
  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  acc4 <- eval_accuracy(clf_model,dsl$test_ds,dev)
  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  acc5 <- eval_accuracy(clf_model,dsl$test_ds,dev)
  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  acc6 <- eval_accuracy(clf_model,dsl$test_ds,dev)

  expect_true(acc1<=acc2)
  expect_true(acc2<=acc4)
  expect_true(acc3<=acc4)
  expect_true(acc4<=acc5)
  expect_true(acc5<=acc6)

  expect_equal(acc1,0.3)
  expect_equal(acc2,0.81666667)
  expect_equal(acc3,0.81666667)
  expect_equal(acc4,0.81666667)
  expect_equal(acc5,0.81666667)
  expect_equal(acc6,0.85)

  clf_model <- create_model(train_x,train_y, c(8,8,8), dimensions = 2)
  clf_model$to(device = dev)

  pretrain_net(1, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  stp1 <- calc_STP(clf_model,dsl$test_ds,sensitive_test,dev)
  pretrain_net(5, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  stp2 <- calc_STP(clf_model,dsl$test_ds,sensitive_test,dev)
  pretrain_net(15, clf_model, dsl, model_type = 1, 0.001,sensitive_test, dev)
  stp3 <- calc_STP(clf_model,dsl$test_ds,sensitive_test,dev)

  expect_equal(stp1,1)
  expect_equal(stp2,0.41558442)
  expect_equal(stp3,0.72727273)
})
