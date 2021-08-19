test_that("test-train_PAN", {

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  setwd("..")

  model1 <- torch::torch_load("./zzz/preclf")
  model2 <- torch::torch_load("./zzz/clf1")
  model3 <- torch::torch_load("./zzz/clf2")
  model11 <- torch::torch_load("./zzz/preadv")
  model22 <- torch::torch_load("./zzz/adv1")
  model33 <- torch::torch_load("./zzz/adv2")

  model1_optimizer_dict <- torch::torch_load("./zzz/preclf_optimizer")
  model2_optimizer_dict <- torch::torch_load("./zzz/clf_optimizer1")
  model3_optimizer_dict <- torch::torch_load("./zzz/clf_optimizer2")
  model11_optimizer_dict <- torch::torch_load("./zzz/preadv_optimizer")
  model22_optimizer_dict <- torch::torch_load("./zzz/adv_optimizer1")
  model33_optimizer_dict <- torch::torch_load("./zzz/adv_optimizer2")

  model1_optimizer <- torch::optim_adam(model1$parameters, lr = 0.001)
  model2_optimizer <- torch::optim_adam(model2$parameters, lr = 0.001)
  model3_optimizer <- torch::optim_adam(model3$parameters, lr = 0.001)
  model11_optimizer <- torch::optim_adam(model11$parameters, lr = 0.001)
  model22_optimizer <- torch::optim_adam(model22$parameters, lr = 0.001)
  model33_optimizer <- torch::optim_adam(model33$parameters, lr = 0.001)

  model1_optimizer$load_state_dict(model1_optimizer_dict)
  model2_optimizer$load_state_dict(model2_optimizer_dict)
  model3_optimizer$load_state_dict(model3_optimizer_dict)
  model11_optimizer$load_state_dict(model11_optimizer_dict)
  model22_optimizer$load_state_dict(model22_optimizer_dict)
  model33_optimizer$load_state_dict(model33_optimizer_dict)

  processed <- torch::torch_load("./zzz/processed")

  dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
                        processed$test_y, batch_size=5, dev=dev)

  monitoring1 <- train_PAN(
    n_ep_pan = 1,
    dsl = dsl,
    clf_model = model1,
    adv_model = model11,
    clf_optimizer = model1_optimizer,
    adv_optimizer = model11_optimizer,
    dev = dev,
    sensitive_train = processed$sensitive_train,
    sensitive_test = processed$sensitive_test,
    batch_size = 5,
    learning_rate_adv = 0.001,
    learning_rate_clf = 0.001,
    lambda = 150,
    verbose = FALSE,
    monitor = TRUE
  )

  expect_true(round(monitoring1$STP[1], 7) == 0.3192681)
  expect_true(round(monitoring1$adversary_acc[1], 7) == 0.5985714)
  expect_true(round(monitoring1$classifier_acc[1], 7) == 0.8381963)
  expect_true(round(monitoring1$adversary_losses[1], 5) == 99.82896)

  monitoring2 <- train_PAN(
    n_ep_pan = 1,
    dsl = dsl,
    clf_model = model2,
    adv_model = model22,
    clf_optimizer = model2_optimizer,
    adv_optimizer = model22_optimizer,
    dev = dev,
    sensitive_train = processed$sensitive_train,
    sensitive_test = processed$sensitive_test,
    batch_size = 5,
    learning_rate_adv = 0.001,
    learning_rate_clf = 0.001,
    lambda = 150,
    verbose = FALSE,
    monitor = TRUE
  )

  expect_true(round(monitoring2$STP[1], 7) == 0.2862404)
  expect_true(round(monitoring2$adversary_acc[1], 7) == 0.6014286)
  expect_true(round(monitoring2$classifier_acc[1], 7) == 0.8408488)
  expect_true(round(monitoring2$adversary_losses[1], 5) == 99.61892)

  monitoring3 <- train_PAN(
    n_ep_pan = 1,
    dsl = dsl,
    clf_model = model3,
    adv_model = model33,
    clf_optimizer = model3_optimizer,
    adv_optimizer = model33_optimizer,
    dev = dev,
    sensitive_train = processed$sensitive_train,
    sensitive_test = processed$sensitive_test,
    batch_size = 5,
    learning_rate_adv = 0.001,
    learning_rate_clf = 0.001,
    lambda = 150,
    verbose = FALSE,
    monitor = TRUE
  )

  expect_true(round(monitoring3$STP[1], 7) == 0.3557559)
  expect_true(round(monitoring3$adversary_acc[1], 7) == 0.6157143)
  expect_true(round(monitoring3$classifier_acc[1], 7) == 0.8381963)
  expect_true(round(monitoring3$adversary_losses[1], 4) == 100.0369)
})
