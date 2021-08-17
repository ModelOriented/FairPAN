test_that("test-plot_that", {

  monitor2 <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/monitoring2")
  monitor3 <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/monitoring3")

  plot1 <- plot_monitor(monitor2$STP ,monitor2$adversary_acc,
                        monitor2$classifier_acc, monitor2$adversary_losses)
  plot2 <- plot_monitor(monitor3$STP ,monitor3$adversary_acc,
                        monitor3$classifier_acc, monitor3$adversary_losses)

  dev   <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  data("adult")
  processed <- preprocess( adult, "salary", "sex", "Male", "Female", c("race"),
                           sample = 0.8, train_size = 0.65, test_size = 0.35,
                           validation_size = 0, seed = 7)

  model1    <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf1")

  dsl       <- dataset_loader(processed$train_x, processed$train_y,
                              processed$test_x, processed$test_y,
                              batch_size=5, dev=dev)

  exp <- Single_explainer(
    processed$test_y,
    model1,
    "classifier",
    processed$data_test,
    processed$data_scaled_test,
    processed$protected_test,
    privileged = "Male",
    batch_size = 5,
    dev = dev,
    verbose = FALSE
  )

  plot3 <- plot_fairness(exp)
  expect_s3_class(plot1, "ggplot")
  expect_s3_class(plot2, "ggplot")
  expect_s3_class(plot3, "ggplot")

})
