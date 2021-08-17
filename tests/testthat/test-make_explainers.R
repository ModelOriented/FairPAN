test_that("test-make_explainers", {
  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  data("adult")

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

  model1 <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf1")
  model2 <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf2")
  model3 <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/clf3")

  dsl <- dataset_loader(processed$train_x, processed$train_y,
                        processed$test_x, processed$test_y,
                        batch_size=5, dev=dev)

  single <- Single_explainer(
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

  dual <- Dual_explainer(
      processed$test_y,
      model1,
      model2,
      "classifier",
      "classifier2",
      processed$data_test,
      processed$data_scaled_test,
      protected = processed$protected_test,
      privileged = "Male",
      batch_size = 5,
      dev = dev,
      verbose = FALSE
    )

  # triple <- Triple_explainer(
  #     processed$test_y,
  #     model1,
  #     model2,
  #     model3,
  #     "classifier",
  #     "classifier2",
  #     "classifier3",
  #     processed$data_test,
  #     processed$data_scaled_test,
  #     protected = processed$protected_test,
  #     privileged = "Male",
  #     batch_size = 5,
  #     dev = dev,
  #     verbose = FALSE
  #   )

  group_metric_matrix <- matrix(0, nrow = 1 , ncol = 12)
  colnames(group_metric_matrix) <- c("TPR","TNR","PPV","NPV","FNR","FPR",
                                     "FDR","FOR","TS","STP","ACC","F1")
  group_metric_matrix[1,] <- c(0.758752,0.022506,0.442064,0.137791,0.285749,
                               0.702997,0.833945,0.81922 ,0.821242,1.307578,
                               0.121692,0.665424)

  group_metric_matrix2 <- matrix(0, nrow = 2 , ncol = 12)
  colnames(group_metric_matrix2) <- c("TPR","TNR","PPV","NPV","FNR","FPR",
                                      "FDR","FOR","TS","STP","ACC","F1")
  group_metric_matrix[1,] <- c(0.758752,0.022506,0.442064,0.137791,0.285749,
                               0.702997, 0.833945,0.81922 ,0.821242,1.307578,
                               0.121692,0.665424)
  group_metric_matrix2[2,] <- c(0.613104,0.026764,0.466572,0.128634,0.301585,
                                0.594817,0.771970,0.816799,0.713992,1.137422,
                                0.112090,0.562148)

  # group_metric_matrix3 <- matrix(0, nrow = 3 , ncol = 12)
  # colnames(group_metric_matrix3) <- c("TPR","TNR","PPV","NPV","FNR","FPR",
  # "FDR","FOR","TS","STP","ACC","F1")
  # group_metric_matrix[1,] <- c(0.758752,0.022506,0.442064,0.137791,0.285749,
  # 0.702997,
  #                              0.833945,0.81922 ,0.821242,1.307578,0.121692,
  #                              0.665424)
  # group_metric_matrix2[2,] <- c(0.613104,0.026764,0.466572,0.128634,0.301585,
  # 0.594817,
  #                               0.771970,0.816799,0.713992,1.137422,0.112090,
  #                               0.562148)
  # group_metric_matrix3[3,] <- c(0.390688,0.029487,0.532692,0.118433,0.288045,
  # 0.383949,
  #                               0.694300,0.843263,0.594714,0.848885,0.096848,
  #                               0.452466)

  expect_true(single$epsilon == 0.8)
  expect_true(single$privileged == "Male")
  expect_true(
    sum(single$protected == processed$protected_test) ==
      length(processed$protected_test)
  )
  expect_true(single$label == "classifier")
  expect_true(
    max( group_metric_matrix - round(single$parity_loss_metric_data, 6)) ==
    min(group_metric_matrix - round(single$parity_loss_metric_data, 6)))


  expect_true(dual$epsilon == 0.8)
  expect_true(dual$privileged == "Male")
  expect_true(sum(dual$protected == processed$protected_test) ==
                length(processed$protected_test))
  expect_true(sum(dual$label == c("classifier", "classifier2")) == 2)

  helper <- (group_metric_matrix2-round(dual$parity_loss_metric_data, 6)) ==
    (group_metric_matrix2-round(dual$parity_loss_metric_data, 6))

  expect_true( sum(helper) == 24)

  # expect_true(triple$epsilon==0.8)
  # expect_true(triple$privileged=="Male")
  # expect_true(sum(triple$protected==processed$protected_test)==
  # length(processed$protected_test))
  # expect_true(sum(triple$label==
  # c("classifier","classifier2","classifier3"))==3)
  #
  # helper <- (group_metric_matrix3-round(triple$parity_loss_metric_data,6))==
  #   (group_metric_matrix3-round(triple$parity_loss_metric_data,6))
  #
  # expect_true( sum(helper) == 36)


})
