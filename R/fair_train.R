fair_train <- function(){
  train_PAN(n_ep_pan, dsl, clf_model, adv_model, dev, sensitive_train,
            sensitive_test, batch_size, learning_rate_adv,
            learning_rate_clf, lambda)

}
