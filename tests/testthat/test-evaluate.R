test_that("test-evaluate", {

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  model1    <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/preclf")
  model2    <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf1")
  model3    <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf2")
  model4     <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf3")
  processed <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/processed")
  dsl       <- dataset_loader(processed$train_x, processed$train_y,
                              processed$test_x, processed$test_y,
                              batch_size = 5, dev = dev)

  acc1 <- eval_accuracy(model1, dsl$test_ds, dev)
  acc2 <- eval_accuracy(model2, dsl$test_ds, dev)
  acc3 <- eval_accuracy(model3, dsl$test_ds, dev)
  acc4 <- eval_accuracy(model4, dsl$test_ds, dev)

  expect_equal(round(acc1, 7), 0.8355438)
  expect_equal(round(acc2, 7), 0.8381963)
  expect_equal(round(acc3, 7), 0.8381963)
  expect_equal(round(acc4, 7), 0.8355438)

  stp1 <- calc_STP(model1, dsl$test_ds, processed$sensitive_test, dev)
  stp2 <- calc_STP(model2, dsl$test_ds, processed$sensitive_test, dev)
  stp3 <- calc_STP(model3, dsl$test_ds, processed$sensitive_test, dev)
  stp4 <- calc_STP(model4, dsl$test_ds, processed$sensitive_test, dev)

  expect_equal(round(stp1, 7), 0.3320388)
  expect_equal(round(stp2, 7), 0.3192681)
  expect_equal(round(stp3, 7), 0.3270079)
  expect_equal(round(stp4, 7), 0.3942961)

})
