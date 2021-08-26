test_that("test-predict", {
  dev       <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"
  setwd("..")
  model1    <- torch::torch_load("./zzz/preclf")
  model2    <- torch::torch_load("./zzz/clf1")
  model3    <- torch::torch_load("./zzz/clf2")
  model4    <- torch::torch_load("./zzz/clf3")
  processed <- torch::torch_load("./zzz/processed")

  dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
                        processed$test_y, batch_size=5, dev=dev)

  preds1   <- make_preds(model1,dsl$test_ds,dev)
  p_preds1 <- make_preds_prob(model1,dsl$test_ds,dev)

  preds2   <- make_preds(model2,dsl$test_ds,dev)
  p_preds2 <- make_preds_prob(model2,dsl$test_ds,dev)

  preds3   <- make_preds(model3,dsl$test_ds,dev)
  p_preds3 <- make_preds_prob(model3,dsl$test_ds,dev)

  preds4   <- make_preds(model4,dsl$test_ds,dev)
  p_preds4 <- make_preds_prob(model4,dsl$test_ds,dev)

  p1 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1)
  p2 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1)
  p3 <- c(2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1)
  p4 <- c(2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1)

  expect_true(min(preds1[1:20] - p1) == max(preds1[1:20] - p1))
  expect_true(min(preds2[1:20] - p2) == max(preds2[1:20] - p2))
  expect_true(min(preds3[1:20] - p3) == max(preds3[1:20] - p3))
  expect_true(min(preds4[1:20] - p4) == max(preds4[1:20] - p4))

  pp1 <- c(0.5652781 , 0.9431339 , 0.6524199 , 0.9998091 , 0.8179238)
  pp2 <- c(0.5501594 , 0.9316815 , 0.6178529 , 0.9997563 , 0.7736207)
  pp3 <- c(0.4990120 , 0.8970793 , 0.5390501 , 0.9995419 , 0.6505134)
  pp4 <- c(0.4344583 , 0.8419650 , 0.4675517 , 0.9989667 , 0.4975611)

  expect_true(round(min(p_preds1[1:5] - pp1), 7) ==
                round(max(p_preds1[1:5] - pp1), 7))
  expect_true(round(min(p_preds2[1:5] - pp2), 7) ==
                round(max(p_preds2[1:5] - pp2), 7))
  expect_true(round(min(p_preds3[1:5] - pp3), 5) ==
                round(max(p_preds3[1:5] - pp3), 5))
  expect_true(round(min(p_preds4[1:5] - pp4), 7) ==
                round(max(p_preds4[1:5] - pp4), 7))
  # not a closure
  expect_error(
    make_preds(7,dsl$test_ds,dev)
  )
  # not an environment
  expect_error(
    make_preds(model1,7,dev)
  )
  # not correct dev
  expect_error(
    make_preds(model1,dsl$test_ds,"GGPPUU")
  )

  # not a closure
  expect_error(
    make_preds_prob(7,dsl$test_ds,dev)
  )
  # not an environment
  expect_error(
    make_preds_prob(model1,7,dev)
  )
  # not correct dev
  expect_error(
    make_preds_prob(model1,dsl$test_ds,"GGPPUU")
  )

})
