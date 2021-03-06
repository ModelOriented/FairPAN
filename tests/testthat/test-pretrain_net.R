test_that("test-pretrain_net", {

  adult               <- fairmodels::adult

  adult               <- adult[1:200, ]
  sensitive           <- adult$sex
  sensitive           <- as.integer(sensitive)

  data                <- adult[, c(2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15)]
  data$workclass      <- as.integer(data$workclass)
  data$education      <- as.integer(data$education)
  data$marital_status <- as.integer(data$marital_status)
  data$occupation     <- as.integer(data$occupation)
  data$relationship   <- as.integer(data$relationship)
  data$native_country <- as.integer(data$native_country)

  data_matrix         <- matrix(unlist(data), ncol = 12)
  data_scaled         <- scale(data_matrix, center = TRUE, scale = TRUE)
  target              <- adult$salary
  target              <- as.integer(target)
  set.seed(7)
  train_indices       <- sample(1:nrow(adult), 0.7 * nrow(adult))
  adult_test          <- adult[setdiff(1:nrow(adult), train_indices), ]
  data_scaled_test    <- data_scaled[setdiff(1:nrow(adult), train_indices), ]

  train_x             <- data_scaled[train_indices, ]
  test_x              <- data_scaled[setdiff(1:nrow(data_scaled),
                                             train_indices),]
  train_y             <- target[train_indices]
  test_y              <- target[setdiff(1:length(target), train_indices)]

  sensitive_train     <- sensitive[train_indices]
  sensitive_test      <- sensitive[setdiff(1:length(sensitive),
                                           train_indices)]

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(train_x, train_y, test_x, test_y, 5, dev)

  clf_model <- create_model(train_x, train_y, c(8, 8, 8), dimensions = 2)
  clf_model$to(device = dev)

  loss1 <- pretrain_net(3, clf_model, dsl, model_type = 1, 0.001,
                        sensitive_test, dev,verbose = FALSE, monitor = FALSE)
  loss2 <- pretrain_net(3, clf_model, dsl, model_type = 2, 0.001,
                        sensitive_test, dev,verbose = FALSE, monitor = FALSE)
  loss3 <- pretrain_net(3, clf_model, dsl, model_type = 0, 0.001,
                        sensitive_test, dev,verbose = FALSE, monitor = FALSE)

  expect_false(loss1$test_loss == loss2$test_loss)
  expect_false(loss2$test_loss == loss3$test_loss)
  expect_false(loss3$test_loss == loss1$test_loss)

  expect_false(loss1$train_loss == loss2$train_loss)
  expect_false(loss2$train_loss == loss3$train_loss)
  expect_false(loss3$train_loss == loss1$train_loss)
  # wrong epoch
  expect_error(
    pretrain_net(
      n_epochs = 3.5,
      model = clf_model,
      dsl = dsl,
      model_type = 1,
      learning_rate = 0.001,
      sensitive_test = sensitive_test,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE)
  )
  # wrong model
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = 7,
      dsl = dsl,
      model_type = 1,
      learning_rate = 0.001,
      sensitive_test = sensitive_test,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE)
  )
  # not a dsl
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = clf_model,
      dsl = 7,
      model_type = 1,
      learning_rate = 0.001,
      sensitive_test = sensitive_test,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE)
  )
  # wrong LR
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = clf_model,
      dsl = dsl,
      model_type = 1,
      learning_rate = 1.5,
      sensitive_test = sensitive_test,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE)
  )
  # wrong dev
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = clf_model,
      dsl = dsl,
      model_type = 1,
      learning_rate = 0.001,
      sensitive_test = sensitive_test,
      dev = "GGPPUU",
      verbose = FALSE,
      monitor = FALSE)
  )
  # sens test not a vector
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = clf_model,
      dsl = dsl,
      model_type = 1,
      learning_rate = 0.001,
      sensitive_test = as.matrix(sensitive_test),
      dev = dev,
      verbose = FALSE,
      monitor = FALSE)
  )
  # monitor not logical
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = clf_model,
      dsl = dsl,
      model_type = 1,
      learning_rate = 0.001,
      sensitive_test = sensitive_test,
      dev = dev,
      verbose = 7,
      monitor = FALSE)
  )
  # wrong model type
  expect_error(
    pretrain_net(
      n_epochs = 3,
      model = clf_model,
      dsl = dsl,
      model_type = 7,
      learning_rate = 0.001,
      sensitive_test = sensitive_test,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE)
  )

})
