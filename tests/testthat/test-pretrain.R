test_that("test-pretrain", {
  data("adult")

  processed <- preprocess(adult,"salary","sex","Male","Female",c("race"),sample=0.05,
                          train_size=0.65,test_size=0.35,validation_size=0,seed=7)

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x, processed$test_y,
                        batch_size=5, dev=dev)

  models <- pretrain(train_x = processed$train_x,train_y = processed$train_y,sensitive_train =
                       processed$sensitive_train, sensitive_test = processed$sensitive_test,batch_size=5,
                     partition=0.65,neurons_clf=c(32,32,32),neurons_adv=c(32,32,32),dimension_clf=2,
                     dimension_adv=1,learning_rate_clf=0.001,learning_rate_adv=0.001,n_ep_preclf=3,
                     n_ep_preadv=5,dsl=dsl,dev=dev,verbose=TRUE,monitor=TRUE)

  eval_accuracy(models$clf_model,dsl$test_ds,dev)
  expect_true(typeof(models$adv_model)=="closure")
  expect_true(typeof(models$clf_optimizer)=="environment")
  expect_true(typeof(models$adv_optimizer)=="environment")

  #dodaj testy odnośnie modeli samorobnych (ale zrobionych przez pretrain net itd)
})
