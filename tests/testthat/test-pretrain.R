test_that("test-pretrain", {

  setwd("..")

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

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
                        processed$test_y, batch_size = 5, dev = dev)

  # Both models created with our package
  models <- pretrain(
    train_x = processed$train_x,
    train_y = processed$train_y,
    sensitive_train = processed$sensitive_train,
    sensitive_test = processed$sensitive_test,
    batch_size = 5,
    partition = 0.65,
    neurons_clf = c(32, 32, 32),
    neurons_adv = c(32, 32, 32),
    dimension_clf = 2,
    dimension_adv = 1,
    learning_rate_clf = 0.001,
    learning_rate_adv = 0.001,
    n_ep_preclf = 1,
    n_ep_preadv = 1,
    dsl = dsl,
    dev = dev,
    verbose = FALSE,
    monitor = FALSE
  )

  acc1 <- eval_accuracy(models$clf_model, dsl$test_ds, dev)
  expect_true(round(acc1, 6) == 0.806366)
  expect_true(typeof(models$adv_model) == "closure")
  expect_true(typeof(models$clf_optimizer) == "environment")
  expect_true(typeof(models$adv_optimizer) == "environment")

  clf                 <- torch::torch_load("./zzz/clf2")
  clf_optimizer_state <- torch::torch_load("./zzz/clf_optimizer2")
  clf_optimizer       <- torch::optim_adam(clf$parameters, lr = 0.001)
  acc2                <- eval_accuracy(clf, dsl$test_ds, dev)
  clf_optimizer$load_state_dict(clf_optimizer_state)

  # Clf provided and pretrained
  models2 <- pretrain(
    clf_model = clf,
    clf_optimizer = clf_optimizer,
    trained = TRUE,
    train_x = processed$train_x,
    train_y = processed$train_y,
    sensitive_train = processed$sensitive_train,
    sensitive_test = processed$sensitive_test,
    batch_size = 5,
    partition = 0.65,
    neurons_clf = c(32, 32, 32),
    neurons_adv = c(32, 32, 32),
    dimension_clf = 2,
    dimension_adv = 1,
    learning_rate_clf = 0.001,
    learning_rate_adv = 0.001,
    n_ep_preclf = 1,
    n_ep_preadv = 1,
    dsl = dsl,
    dev = dev,
    verbose = FALSE,
    monitor = FALSE
  )

  acc3 <- eval_accuracy(models2$clf_model, dsl$test_ds, dev)
  expect_true(acc2 == acc3)
  expect_true(round(acc3, 7) == 0.8381963)
  expect_true(typeof(models2$adv_model) == "closure")
  expect_true(typeof(models2$clf_optimizer) == "environment")
  expect_true(typeof(models2$adv_optimizer) == "environment")

  # Clf architecture provided but not trained

  clf2 <-
    create_model(processed$train_x, processed$train_y, c(4, 4), 2)

  models3 <- pretrain(
    clf_model = clf2,
    trained = FALSE,
    train_x = processed$train_x,
    train_y = processed$train_y,
    sensitive_train = processed$sensitive_train,
    sensitive_test = processed$sensitive_test,
    batch_size = 5,
    partition = 0.65,
    neurons_clf = c(32, 32, 32),
    neurons_adv = c(32, 32, 32),
    dimension_clf = 2,
    dimension_adv = 1,
    learning_rate_clf = 0.001,
    learning_rate_adv = 0.001,
    n_ep_preclf = 1,
    n_ep_preadv = 1,
    dsl = dsl,
    dev = dev,
    verbose = FALSE,
    monitor = FALSE
  )

  acc4 <- eval_accuracy(models3$clf_model, dsl$test_ds, dev)
  expect_true(acc4 != acc3)
  expect_true(round(acc4, 7) == 0.806366)
  expect_true(typeof(models3$adv_model) == "closure")
  expect_true(typeof(models3$clf_optimizer) == "environment")
  expect_true(typeof(models3$adv_optimizer) == "environment")

  # n ep preclf wrong
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = - 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # n ep preadv wrong
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = - 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # batch size not integer
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5.5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # seed not integer
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE,
      seed = 7.5
    )
  )
  # dsl not dsl
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = 7,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # LR clf over 1
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 1.5,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # LR adv over 1
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 1.5,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # partition over 1
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 1.5,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # dev wrong
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = "GGPPUU",
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # sensitive test not a vector
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = as.matrix(processed$sensitive_test),
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # sensitive train not a vector
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = as.matrix(processed$sensitive_train),
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # trained not logical
  expect_error(
    pretrain(
      trained = 7,
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # verbose not logical
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = 7,
      monitor = FALSE
    )
  )
  # dim clf wrong
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 7,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # dim adv wrong
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 7,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # neurons clf not an integer
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32.5, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # neurons adv not an integer
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32.5, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # clf optimizer not an optimizer
  expect_error(
    pretrain(
      clf_optimizer = 7,
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # train y not a vector
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = as.matrix(processed$train_y),
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # train_x not a matrix
  expect_error(
    pretrain(
      train_x = as.vector(processed$train_x),
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # not equal lengths of train x and y
  expect_error(
    pretrain(
      train_x = processed$train_x,
      train_y = processed$train_y[1:4],
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )
  # clf not a model
  expect_error(
    pretrain(
      clf_model = 7,
      train_x = processed$train_x,
      train_y = processed$train_y,
      sensitive_train = processed$sensitive_train,
      sensitive_test = processed$sensitive_test,
      batch_size = 5,
      partition = 0.65,
      neurons_clf = c(32, 32, 32),
      neurons_adv = c(32, 32, 32),
      dimension_clf = 2,
      dimension_adv = 1,
      learning_rate_clf = 0.001,
      learning_rate_adv = 0.001,
      n_ep_preclf = 1,
      n_ep_preadv = 1,
      dsl = dsl,
      dev = dev,
      verbose = FALSE,
      monitor = FALSE
    )
  )

})
