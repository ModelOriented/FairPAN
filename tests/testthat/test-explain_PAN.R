test_that("test-explainPAN", {
  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  adult <- fairmodels::adult

  processed <- preprocess(
      adult,
      "salary",
      "sex",
      "Male",
      "Female",
      c("race"),
      sample = 0.8,
      train_size = 0.65,
      test_size = 0.35,
      validation_size = 0,
      seed = 7
    )

  setwd("..")
  model1 <- torch::torch_load("./zzz/clf1")

  dsl <- dataset_loader(processed$train_x, processed$train_y,
                        processed$test_x, processed$test_y,
                        batch_size=5, dev=dev)

  explainer <- explain_PAN(
    processed$test_y,
    model1,
    "classifier",
    processed$data_test,
    processed$data_scaled_test,
    batch_size = 5,
    dev = dev,
    verbose = FALSE
  )

  data <- list("age"=24,"fnlwgt"=201603,"education_num"=9,"hours_per_week"=30)
  y <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
  y_hat <- c(0.0077336240, 0.6553712487, 0.0014575135, 0.0001699024,
             0.0005531592)
  residuals <- c(-0.0077336240, -0.6553712487, -0.0014575135, -0.0001699024,
                 -0.0005531592)

  expect_true(4 == sum(explainer$data[1, c(2, 4, 6, 14)] == data))
  expect_true(sum(y == explainer$y[1:10]) == 10)
  expect_true(sum(round(explainer$y_hat[1:5], 5) == round(y_hat, 5)) == 5)
  expect_true(sum(round(explainer$residuals[1:5], 5) == round(residuals,5 )) == 5)
  expect_true(sum(explainer$class == c("net", "nn_module")) == 2)
  expect_true(explainer$label == "classifier")
  expect_true(is.null(explainer$weights))

  # not vector
  expect_error(
    explainer <- explain_PAN(
      as.matrix(processed$test_y),
      model1,
      "classifier",
      processed$data_test,
      processed$data_scaled_test,
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not closure
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      7,
      "classifier",
      processed$data_test,
      processed$data_scaled_test,
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not character
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      model1,
      7,
      processed$data_test,
      processed$data_scaled_test,
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not list
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      model1,
      "classifier",
      as.matrix(processed$data_test),
      processed$data_scaled_test,
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not matrix
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      model1,
      "classifier",
      processed$data_test,
      as.list(processed$data_scaled_test),
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not equal rows
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      model1,
      "classifier",
      as.matrix(processed$data_test, nrow = 2),
      processed$data_scaled_test,
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not integer
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      model1,
      "classifier",
      processed$data_test,
      processed$data_scaled_test,
      batch_size = 0.5,
      dev = dev,
      verbose = FALSE
    )
  )
  # not dev type
  expect_error(
    explainer <- explain_PAN(
      processed$test_y,
      model1,
      "classifier",
      processed$data_test,
      processed$data_scaled_test,
      batch_size = 5,
      dev = "GGPPUU",
      verbose = FALSE
    )
  )
  # not logical
  expect_error(
      explainer <- explain_PAN(
      processed$test_y,
      model1,
      "classifier",
      processed$data_test,
      processed$data_scaled_test,
      batch_size = 5,
      dev = dev,
      verbose = 7
    )
  )


})
