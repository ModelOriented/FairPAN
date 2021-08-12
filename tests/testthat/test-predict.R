test_that("test-predict", {
  data("adult")
  adult<-adult[1:600,]
  adult
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

  sil<-pretrain_net(5, clf_model, dsl, model_type = 1, 0.001, sensitive_test, dev,verbose = FALSE,
               monitor = FALSE)
  preds1 <- make_preds(clf_model,dsl$test_ds,dev)
  p_preds1 <- make_preds_prob(clf_model,dsl$test_ds,dev)

  sil<-pretrain_net(5, clf_model, dsl, model_type = 1, 0.001, sensitive_test, dev,verbose = FALSE,
               monitor = FALSE)
  preds2 <- make_preds(clf_model,dsl$test_ds,dev)
  p_preds2 <- make_preds_prob(clf_model,dsl$test_ds,dev)

  sil<-pretrain_net(5, clf_model, dsl, model_type = 1, 0.001, sensitive_test, dev,verbose = FALSE,
               monitor = FALSE)
  preds3 <- make_preds(clf_model,dsl$test_ds,dev)
  p_preds3 <- make_preds_prob(clf_model,dsl$test_ds,dev)

  sil<-pretrain_net(5, clf_model, dsl, model_type = 1, 0.001, sensitive_test, dev,verbose = FALSE,
               monitor = FALSE)
  preds4 <- make_preds(clf_model,dsl$test_ds,dev)
  p_preds4 <- make_preds_prob(clf_model,dsl$test_ds,dev)

  sil<-pretrain_net(5, clf_model, dsl, model_type = 1, 0.001, sensitive_test, dev,verbose = FALSE,
               monitor = FALSE)
  preds5 <- make_preds(clf_model,dsl$test_ds,dev)
  p_preds5 <- make_preds_prob(clf_model,dsl$test_ds,dev)

  sil<-pretrain_net(5, clf_model, dsl, model_type = 1, 0.001, sensitive_test, dev,verbose = FALSE,
               monitor = FALSE)
  preds6 <- make_preds(clf_model,dsl$test_ds,dev)
  p_preds6 <- make_preds_prob(clf_model,dsl$test_ds,dev)

  p1 <- c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
  p2 <- c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
  p3 <- c(1,2,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1)
  p4 <- c(1,2,1,1,1,2,1,1,2,1,1,1,1,1,1,2,1,1,1,1)
  p5 <- c(1,2,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1)
  p6 <- c(1,2,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1)

  expect_true(min(preds1[1:20]-p1)==max(preds1[1:20]-p1))
  expect_true(min(preds2[1:20]-p2)==max(preds2[1:20]-p2))
  expect_true(min(preds3[1:20]-p3)==max(preds3[1:20]-p3))
  expect_true(min(preds4[1:20]-p4)==max(preds4[1:20]-p4))
  expect_true(min(preds5[1:20]-p5)==max(preds5[1:20]-p5))
  expect_true(min(preds6[1:20]-p6)==max(preds6[1:20]-p6))

  pp1 <- c(0.9742318, 0.8315669, 0.9101276, 0.9913149, 0.7777176)
  pp2 <- c(0.9923260, 0.7124295, 0.9163929, 0.9993702, 0.7051284)
  pp3 <- c(0.9993027, 0.4631852, 0.9245963, 0.9999706, 0.6939899)
  pp4 <- c(0.9999322, 0.2948388, 0.9238539, 0.9999983, 0.7891934)
  pp5 <- c(0.9999931, 0.2633764, 0.9630195, 0.9999999, 0.8948784)
  pp6 <- c(0.9999993, 0.2399016, 0.9830412, 1.0000000, 0.9429768)

  expect_true(round(min(p_preds1[1:5]-pp1),7)==round(max(p_preds1[1:5]-pp1),7))
  expect_true(round(min(p_preds2[1:5]-pp2),7)==round(max(p_preds2[1:5]-pp2),7))
  expect_true(round(min(p_preds3[1:5]-pp3),7)==round(max(p_preds3[1:5]-pp3),7))
  expect_true(round(min(p_preds4[1:5]-pp4),7)==round(max(p_preds4[1:5]-pp4),7))
  expect_true(round(min(p_preds5[1:5]-pp5),7)==round(max(p_preds5[1:5]-pp5),7))
  expect_true(round(min(p_preds6[1:5]-pp6),7)==round(max(p_preds6[1:5]-pp6),7))

})
