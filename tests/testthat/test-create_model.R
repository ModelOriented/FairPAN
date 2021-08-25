test_that("test-create_model", {

  train_x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  train_y <- c(1, 2, 3)
  model   <- create_model(train_x,
                 train_y,
                 neurons = c(16, 8, 16),
                 dimensions = 1)
  model2  <-  create_model(train_x,
                 train_y,
                 neurons = c(16, 8, 16),
                 dimensions = 1)
  model3  <- create_model(train_x,
                 train_y,
                 neurons = c(16, 16, 16),
                 dimensions = 1)
  #tests for seed
  expect_true(torch::torch_equal(model$fc1$weight, model2$fc1$weight))
  expect_true(torch::torch_equal(model$fc2$weight, model2$fc2$weight))
  expect_true(torch::torch_equal(model$fc3$weight, model2$fc3$weight))

  expect_true(torch::torch_equal(model$fc1$weight, model3$fc1$weight))
  expect_false(torch::torch_equal(model$fc2$weight, model3$fc2$weight))
  expect_false(torch::torch_equal(model$fc3$weight, model3$fc3$weight))

  expect_equal(typeof(model), "closure")

  expect_error(
    create_model(
      train_x,
      train_y,
      neurons = c(16, 8, 16),
      dimensions = -3
    )
  )

  expect_error(
    create_model(
      train_x,
      as.matrix(train_y),
      neurons = c(16, 8, 16),
      dimensions = 1
    )
  )

  expect_error(
    create_model(
      as.list(train_x),
      train_y,
      neurons = c(16, 8, 16),
      dimensions = 1
    )
  )

  expect_error(
    create_model(
      train_x,
      as.matrix(train_y,nrow = 2),
      neurons = c(16, 8, 16),
      dimensions = 1
    )
  )

  expect_error(
    create_model(
      train_x,
      train_y,
      neurons = c(16, 8, 16),
      dimensions = 0.5
    )
  )


})
