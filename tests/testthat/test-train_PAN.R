test_that("test-train_PAN", {
  data("adult")
  processed <- preprocess(adult,"salary","sex","Male","Female",c("race"),sample=0.05,
                          train_size=0.65,test_size=0.35,validation_size=0,seed=7)

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x, processed$test_y,
                        batch_size=5, dev=dev)

  models <- pretrain(train_x = processed$train_x,train_y = processed$train_y,sensitive_train =
                       processed$sensitive_train, sensitive_test = processed$sensitive_test,batch_size=5,
                     partition=0.7,neurons_clf=c(32,32,32),neurons_adv=c(32,32,32),dimension_clf=2,
                     dimension_adv=1,learning_rate_clf=0.001,learning_rate_adv=0.001,n_ep_preclf=3,
                     n_ep_preadv=5,dsl=dsl,dev=dev,verbose=FALSE,monitor=FALSE)

  monitoring1 <- train_PAN(n_ep_pan=5, dsl=dsl, clf_model=models$clf_model,
                          adv_model=models$adv_model, clf_optimizer=models$clf_optimizer,
                          adv_optimizer=models$adv_optimizer, dev=dev,
                          sensitive_train=processed$sensitive_train,
                          sensitive_test=processed$sensitive_test, batch_size=5,
                          learning_rate_adv=0.001, learning_rate_clf=0.001, lambda=150,
                          verbose=FALSE, monitor=TRUE)

  expect_true(round(monitoring1$STP[1],7)==0.3320388)
  expect_true(round(monitoring1$STP[5],7)==0.3689320)
  expect_true(round(monitoring1$adversary_acc[1],7)==0.5900000)
  expect_true(round(monitoring1$adversary_acc[5],7)==0.6000000)
  expect_true(round(monitoring1$classifier_acc[1],7)==0.8355438)
  expect_true(round(monitoring1$classifier_acc[5],7)==0.8355438)
  expect_true(round(monitoring1$adversary_losses[1],5)==100.08660 )
  expect_true(round(monitoring1$adversary_losses[5],5)==100.60658)

  monitoring2 <- train_PAN(n_ep_pan=5, dsl=dsl, clf_model=models$clf_model,
                           adv_model=models$adv_model, clf_optimizer=models$clf_optimizer,
                           adv_optimizer=models$adv_optimizer, dev=dev,
                           sensitive_train=processed$sensitive_train,
                           sensitive_test=processed$sensitive_test, batch_size=5,
                           learning_rate_adv=0.001, learning_rate_clf=0.001, lambda=150,
                           verbose=FALSE, monitor=TRUE)

  expect_true(round(monitoring2$STP[1],7)==0.3996764)
  expect_true(round(monitoring2$STP[5],7)==0.6744539)
  expect_true(round(monitoring2$adversary_acc[1],7)==0.5928571)
  expect_true(round(monitoring2$adversary_acc[5],7)==0.5785714)
  expect_true(round(monitoring2$classifier_acc[1],7)==0.8328912 )
  expect_true(round(monitoring2$classifier_acc[5],7)==0.8302387)
  expect_true(round(monitoring2$adversary_losses[1],4)==101.0836 )
  expect_true(round(monitoring2$adversary_losses[5],4)==102.6199)

  monitoring3 <- train_PAN(n_ep_pan=3, dsl=dsl, clf_model=models$clf_model,
                           adv_model=models$adv_model, clf_optimizer=models$clf_optimizer,
                           adv_optimizer=models$adv_optimizer, dev=dev,
                           sensitive_train=processed$sensitive_train,
                           sensitive_test=processed$sensitive_test, batch_size=5,
                           learning_rate_adv=0.001, learning_rate_clf=0.001, lambda=150,
                           verbose=FALSE, monitor=TRUE)

  expect_true(round(monitoring3$STP[1],7)==0.7194175 )
  expect_true(round(monitoring3$STP[3],7)==0.7194175)
  expect_true(round(monitoring3$adversary_acc[1],7)==0.5728571 )
  expect_true(round(monitoring3$adversary_acc[3],7)==0.5628571)
  expect_true(round(monitoring3$classifier_acc[1],7)==0.8275862  )
  expect_true(round(monitoring3$classifier_acc[3],7)==0.8275862)
  expect_true(round(monitoring3$adversary_losses[1],4)==102.9340  )
  expect_true(round(monitoring3$adversary_losses[3],4)==103.5985)
})
