#' Proceeds with fair training with PAN model
#'
#' Core function of the whole package. It trains a PAN model with fair predictions.
#'
#' @param train_x numeric, scaled matrix of predictors used for training
#' @param test_x integer, matrix of predictors used for testing
#' @param train_y numeric, scaled vector of target used for training
#' @param test_y integer, vector of predictors used for testing
#' @param sensitive_train integer, vector of sensitive values used for training
#' @param sensitive_test integer, vector of sensitive values used for testing
#' @param data_scaled_test scaled matrix of numerical values representing predictors
#' @param protected numerical vector of sensitive variables
#' @param priviliged string label of privileged class in protected
#' @param data numerical table of predictors
#' @param batch_size integer indicating a batch size used in dataloader. Default: 50
#' @param partition float from [0,1] range setting the size of train vector (test size
#'                  equals 1-partition). Default = 0.7.
#' @param neurons_clf integer vector describing a neural architecture of classifier
#'                    network. Default: c(32,32,32). This notation means that the network
#'                    has 3 layers with 32 neurons each.
#' @param neurons_adv integer vector describing a neural architecture of adversarial
#'                    network. Default: c(32,32,32). This notation means that the network
#'                    has 3 layers with 32 neurons each.
#' @param dimension_clf integer from [0,2] setting nnf_softmax dimension for classifier.
#'                      Default: 2
#' @param dimension_adv integer from [0,2] setting nnf_softmax dimension for adversarial.
#'                      Default: 2
#' @param learning_rate_clf float from [0,1] setting learning rate for classifier.
#'                          Default: 0.001
#' @param learning_rate_adv float from [0,1] setting learning rate for classifier.
#'                          Default: 0.001
#' @param n_ep_preclf integer setting number of epochs for preclassifiers training.
#'                    Default: 5
#' @param n_ep_only integer setting number of epochs for classifier only training.
#'                  Default: 30
#' @param n_ep_preadv integer setting number of epochs for preadversarials training.
#'                    Default : 10
#' @param n_ep_pan integer setting number of epochs for PAN training. Default : 50
#' @param lambda integer parameter regulating learning proccess (intuition: the bigger it is,
#'               the fairer predictions). Default: 50
#' @return NULL
#' @export
#'
#' @examples
fair_train_old <- function(train_x, test_x, train_y, test_y, sensitive_train, sensitive_test,
                      data_scaled_test, protected, priviliged, data, batch_size = 50,
                      partition = 0.7, neurons_clf = c(32,32,32),
                      neurons_adv = c(32,32,32), dimension_clf = 2, dimension_adv = 2,
                      learning_rate_clf = 0.001, learning_rate_adv = 0.001,
                      n_ep_preclf = 5, n_ep_only=30, n_ep_preadv = 10, n_ep_pan = 50,
                      lambda = 50 ){

  dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(train_x, train_y, test_x, test_y, batch_size, dev)

  clf_model <- create_model(train_x,train_y, neurons_clf, dimensions = dimension_clf)
  clf_model$to(device = dev)

  pretrain_net(n_ep_preclf, clf_model, dsl, model_type = 1, learning_rate_clf,
               sensitive_test, dev)

  preds<-make_preds(clf_model, dsl$test_ds, dev)

  p_preds <- make_preds_prob(clf_model, dsl$train_ds, dev)

  eval_accuracy( clf_model, dsl$test_ds, dev)

  #exp1 <-  Single_explainer(test_y,clf_model,data,data_scaled_test,test_y,protected,priviliged,batch_size,dev)

  #plot(exp1)

  clf_only_model <- create_model(train_x,train_y, neurons_clf, dimension_clf)
  clf_only_model$to(device = dev)
  pretrain_net(n_ep_only, clf_only_model, dsl, model_type = 2, learning_rate_clf,
               sensitive_test, dev)

  pre_clf_model <- create_model(train_x,train_y, neurons_clf, dimension_clf)
  pre_clf_model$to(device = dev)
  pretrain_net(n_ep_preclf, pre_clf_model, dsl, model_type = 1, learning_rate_clf,
               sensitive_test, dev)

  prepared_data <- prepare_to_adv(p_preds[,2], sensitive_train, partition)

  dsl_adv <- dataset_loader(prepared_data$train_x,prepared_data$train_y,
                            prepared_data$test_x,prepared_data$test_y,
                            batch_size = batch_size, dev)

  adv_model <- create_model(prepared_data$train_x,prepared_data$train_y,
                            neurons = neurons_adv, dimensions = dimension_adv)

  adv_model$to(device = dev)
  pretrain_net(n_ep_preadv, adv_model, dsl_adv, model_type = 0, learning_rate_adv,
               sensitive_test, dev)

  prepared_data$test_y

  make_preds(adv_model,dsl_adv$test_ds, dev)

  eval_accuracy(adv_model,dsl_adv$test_ds, dev)

  train_PAN(n_ep_pan, dsl, clf_model, adv_model, dev, sensitive_train,
            sensitive_test, batch_size, learning_rate_adv,
            learning_rate_clf, lambda)

 # exp <-  Single_explainer(test_y,clf_model,data,data_scaled_test,test_y,protected,priviliged,batch_size,dev)

  plot(exp)
}
