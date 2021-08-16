# data("adult")
#
# processed <- preprocess(adult,"salary","sex","Male","Female",c("race"),sample=0.05,
#                         train_size=0.65,test_size=0.35,validation_size=0,seed=7)
# torch_save(processed,path = "~/Fairness 2021/FairPAN/tests/zzz/processed")
#
# dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"
# #torch_save(dev,path = "~/Fairness 2021/FairPAN/tests/zzz/dev")
#
# dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x, processed$test_y,
#                       batch_size=5, dev=dev)
# #torch_save(dsl,path = "~/Fairness 2021/FairPAN/tests/zzz/dsl")
#
# models <- pretrain(train_x = processed$train_x,train_y = processed$train_y,sensitive_train =
#                    processed$sensitive_train, sensitive_test = processed$sensitive_test,batch_size=5,
#                    partition=0.65,neurons_clf=c(32,32,32),neurons_adv=c(32,32,32),dimension_clf=2,
#                    dimension_adv=1,learning_rate_clf=0.001,learning_rate_adv=0.001,n_ep_preclf=3,
#                    n_ep_preadv=10,dsl=dsl,dev=dev,verbose=TRUE,monitor=TRUE)
#
# torch_save(models$clf_model,path = "~/Fairness 2021/FairPAN/tests/zzz/preclf")
# torch_save(models$adv_model,path = "~/Fairness 2021/FairPAN/tests/zzz/preadv")
# torch_save(models$clf_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/preclf_optimizer")
# torch_save(models$adv_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/preadv_optimizer")
# # preclf_optimizer_dict <- torch_load("~/Fairness 2021/FairPAN/tests/zzz/preclf_optimizer")
# # models$clf_optimizer$load_state_dict(preclf_optimizer_dict)
# # model1<-torch_load("~/Fairness 2021/FairPAN/tests/zzz/model1")
# # model1
# #
# monitoring1 <- train_PAN(n_ep_pan=1, dsl=dsl, clf_model=models$clf_model,
#                          adv_model=models$adv_model, clf_optimizer=models$clf_optimizer,
#                          adv_optimizer=models$adv_optimizer, dev=dev,
#                          sensitive_train=processed$sensitive_train,
#                          sensitive_test=processed$sensitive_test, batch_size=5,
#                          learning_rate_adv=0.001, learning_rate_clf=0.001, lambda=150,
#                          verbose=TRUE, monitor=TRUE)
# torch_save(models$clf_model,path = "~/Fairness 2021/FairPAN/tests/zzz/clf1")
# torch_save(models$adv_model,path = "~/Fairness 2021/FairPAN/tests/zzz/adv1")
# torch_save(monitoring1,path = "~/Fairness 2021/FairPAN/tests/zzz/monitoring1")
# torch_save(models$clf_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/clf_optimizer1")
# torch_save(models$adv_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/adv_optimizer1")
#
#
# monitoring2 <- train_PAN(n_ep_pan=2, dsl=dsl, clf_model=models$clf_model,
#                          adv_model=models$adv_model, clf_optimizer=models$clf_optimizer,
#                          adv_optimizer=models$adv_optimizer, dev=dev,
#                          sensitive_train=processed$sensitive_train,
#                          sensitive_test=processed$sensitive_test, batch_size=5,
#                          learning_rate_adv=0.001, learning_rate_clf=0.001, lambda=150,
#                          verbose=TRUE, monitor=TRUE)
# torch_save(models$clf_model,path = "~/Fairness 2021/FairPAN/tests/zzz/clf2")
# torch_save(models$adv_model,path = "~/Fairness 2021/FairPAN/tests/zzz/adv2")
# torch_save(monitoring2,path = "~/Fairness 2021/FairPAN/tests/zzz/monitoring2")
# torch_save(models$clf_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/clf_optimizer2")
# torch_save(models$adv_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/adv_optimizer2")
#
# monitoring3 <- train_PAN(n_ep_pan=3, dsl=dsl, clf_model=models$clf_model,
#                          adv_model=models$adv_model, clf_optimizer=models$clf_optimizer,
#                          adv_optimizer=models$adv_optimizer, dev=dev,
#                          sensitive_train=processed$sensitive_train,
#                          sensitive_test=processed$sensitive_test, batch_size=5,
#                          learning_rate_adv=0.001, learning_rate_clf=0.001, lambda=150,
#                          verbose=TRUE, monitor=TRUE)
#
# torch_save(models$clf_model,path = "~/Fairness 2021/FairPAN/tests/zzz/clf3")
# torch_save(models$adv_model,path = "~/Fairness 2021/FairPAN/tests/zzz/adv3")
# torch_save(monitoring3,path = "~/Fairness 2021/FairPAN/tests/zzz/monitoring3")
# torch_save(models$clf_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/clf_optimizer3")
# torch_save(models$adv_optimizer$state_dict(),path = "~/Fairness 2021/FairPAN/tests/zzz/adv_optimizer3")
#
# #zapisać monitoringi do train_PAN
# #zapisać 4 modele do: evaluate, preds, make explainers
