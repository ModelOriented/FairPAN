test_that("test-prepare_data", {

  train_x = matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  train_y = c(1, 2, 3)
  test_x = matrix(c(1, 2, 3, 4), nrow = 2)
  test_y = c(1, 2)


  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"
  dsl <- dataset_loader(train_x,train_y,test_x,test_y,batch_size=1,dev)

  expect_equal(typeof(dsl), "list")
  expect_equal(typeof(dsl$train_ds), "environment")
  expect_equal(typeof(dsl$train_ds$y), "externalptr")

  expect_true(torch::torch_equal(dsl$train_ds$x_cont, torch::torch_tensor(train_x)))
  expect_true(torch::torch_equal(dsl$test_ds$x_cont, torch::torch_tensor(test_x)))
  expect_true(torch::torch_equal(dsl$train_ds$y,
                          torch::torch_tensor(train_y, dtype = torch::torch_long())))
  expect_true(torch::torch_equal(dsl$test_ds$y,
                          torch::torch_tensor(test_y, dtype = torch::torch_long())))

  expect_true(torch::torch_equal(dsl$train_dl$dataset$x_cont, torch::torch_tensor(train_x)))

  expect_true(torch::torch_equal(dsl$test_dl$dataset$x_cont, torch::torch_tensor(test_x)))

  expect_true(torch::torch_equal(
    dsl$train_dl$dataset$y,
    torch::torch_tensor(train_y, dtype = torch::torch_long())
  ))

  expect_true(torch::torch_equal(
    dsl$test_dl$dataset$y,
    torch::torch_tensor(test_y, dtype = torch::torch_long())
  ))
  expect_equal(dsl$train_dl$batch_size, 1)


  preds <- c(0.3454, 0.7746, 0.1414, 0.1321, 0.4788, 0.2038)
  sensitive <- c(2, 2, 1, 1, 2, 1)
  prepared <- prepare_to_adv(preds, sensitive, partition = 0.5)

  expect_equal(sum(preds), sum(c(prepared$train_x, prepared$test_x)))
  expect_equal(sum(sensitive), sum(c(prepared$train_y, prepared$test_y)))
  expect_equal(length(sensitive), length(preds))
  expect_equal(length(prepared$train_x), length(prepared$test_x))
  expect_equal(length(prepared$train_y), length(prepared$test_y))
  expect_true(is.vector(prepared$train_y))
  expect_true(is.matrix(prepared$train_x))
  expect_true(is.matrix(prepared$test_x))
  expect_true(is.vector(prepared$test_y))

  expect_equal(typeof(prepared), "list")


  adult <- fairmodels::adult

  processed <-
    preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )

  expect_true(is.matrix(processed$train_x))
  expect_true(is.vector(processed$train_y))
  expect_true(nrow(processed$train_x) == length(processed$train_y))
  expect_true(is.vector(processed$sensitive_train))
  expect_true(length(processed$train_y) == length(processed$sensitive_train))

  expect_true(is.matrix(processed$test_x))
  expect_true(is.vector(processed$test_y))
  expect_true(nrow(processed$test_x) == length(processed$test_y))
  expect_true(is.vector(processed$sensitive_test))
  expect_true(length(processed$test_y) == length(processed$sensitive_test))


  expect_true(is.matrix(processed$valid_x))
  expect_true(is.vector(processed$valid_y))
  expect_true(nrow(processed$valid_x) == length(processed$valid_y))
  expect_true(is.vector(processed$sensitive_valid))
  expect_true(length(processed$valid_y) == length(processed$sensitive_valid))

  expect_true(is.matrix(processed$data_scaled_test))
  expect_true(is.matrix(processed$data_scaled_valid))

  expect_true(is.list(processed$data_test))
  expect_true(processed$protected_test[1] == "Male")
  expect_true(nrow(processed$data_valid) == 0)
  expect_true(length(processed$protected_valid) == 0)

  expect_error(
    dataset_loader(
      "not numeric",
      train_y,
      test_x,
      test_y,
      batch_size=1,
      dev)
  )

  expect_error(
    dataset_loader(
      train_x,
      "not numeric",
      test_x,
      test_y,
      batch_size=1,
      dev)
  )

  expect_error(
    dataset_loader(
      train_x,
      train_y,
      "not numeric",
      test_y,
      batch_size=1,
      dev)
  )

  expect_error(
    dataset_loader(
      train_x,
      train_y,
      test_x,
      "not numeric",
      batch_size=1,
      dev)
  )

  expect_error(
    dataset_loader(
      train_x,
      train_y,
      test_x,
      test_y,
      batch_size=0.5,
      dev)
  )

  expect_error(
    dataset_loader(
      train_x,
      train_y,
      test_x,
      test_y,
      batch_size=1,
      dev = "GGPPUU")
  )

  expect_error(
    prepare_to_adv(
      as.matrix(preds),
      sensitive,
      partition = 0.5)
  )

  expect_error(
    prepare_to_adv(
      preds,
      as.matrix(sensitive),
      partition = 0.5)
  )

  expect_error(
    prepare_to_adv(
      preds,
      sensitive,
      partition = 1.5)
  )

  expect_error(
    preprocess(
      7,
      "salary",
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      7,
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      7,
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      "sex",
      7,
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      7,
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      "Female",
      c(7),
      sample = 7,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 7,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 1,
      test_size = - 1,
      validation_size = 1,
      seed = 7
    )
  )

  expect_error(
    preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.05,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7.5
    )
  )

  processed2 <- preprocess(
    adult,
    "salary",
    "sex",
    "Male",
    "Female",
    NULL,
    sample = 0.05,
    train_size = 0.65,
    test_size = 0.35,
    validation_size = 0,
    seed = 7
  )

  expect_true(is.matrix(processed2$train_x))



  })
