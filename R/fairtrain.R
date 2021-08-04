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
#' @param BATCH_SIZE integer indicating a batch size used in dataloader. Default: 50
#' @param NEURONS_CLF integer vector describing a neural architecture of classifier
#'                    network. Default: c(32,32,32). This notation means that the network
#'                    has 3 layers with 32 neurons each.
#' @param NEURONS_ADV integer vector describing a neural architecture of adversarial
#'                    network. Default: c(32,32,32). This notation means that the network
#'                    has 3 layers with 32 neurons each.
#' @param DIMENSION_CLF integer from [0,2] setting nnf_softmax dimension for classifier.
#'                      Default: 2
#' @param DMIENSION_ADV integer from [0,2] setting nnf_softmax dimension for adversarial.
#'                      Default: 2
#' @param N_EP_PRECLF integer setting number of epochs for preclassifiers training.
#'                    Default: 5
#' @param LEARNING_RATE_CLF float from [0,1] setting learning rate for classifier.
#'                          Default: 0.001
#' @param PROTECTED numerical vector of sensitive variables
#' @param PRIVILIGED string label of privileged class in protected
#' @param PARTITION float from [0,1] range setting the size of train vector (test size
#'                  equals 1-PARTITION). Default = 0.7.
#' @param N_EP_PAN integer setting number of epochs for PAN training. Default : 50
#' @param LEARNING_RATE_ADV float from [0,1] setting learning rate for classifier.
#'                          Default: 0.001
#' @param LAMBDA integer parameter regulating learning proccess (intuition: the bigger it is,
#'               the fairer predictions). Default: 50
#' @param DATA numerical table of predictors
#' @param N_EP_PREADV integer setting number of epochs for preadversarials training.
#'                    Default : 10
#'
#' @return
#' @export
#'
#' @examples
fairtrain <- function(train_x, test_x, train_y, test_y, sensitive_train, sensitive_test,
                      BATCH_SIZE = 50, NEURONS_CLF = c(32,32,32), NEURONS_ADV = c(32,32,32), DIMENSION_CLF = 2, DMIENSION_ADV = 2,
                      N_EP_PRECLF = 5, LEARNING_RATE_CLF = 0.001, PROTECTED, PRIVILIGED, PARTITION = 0.7,
                      N_EP_PAN = 50, LEARNING_RATE_ADV = 0.001, LAMBDA = 50, DATA, N_EP_PREADV = 10){

  dev <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

  dsl <- dataset_loader(train_x,train_y,test_x,test_y,batch_size = BATCH_SIZE)

  clf_model <- create_model(train_x,train_y, NEURONS_CLF, dimensions = DIMENSION_CLF)
  clf_model$to(device = dev)

  pretrain_net(N_EP_PRECLF, clf_model, dsl, model_type = 1, LEARNING_RATE_CLF,
               sensitive_test)

  preds<-make_preds(clf_model, dsl$test_ds)

  p_preds <- make_preds_prob(clf_model, dsl$train_ds)

  eval_clf( clf_model, dsl$test_ds)

  exp1 <-  GAN_explainer(test_y,clf_model,DATA,PROTECTED,PRIVILIGED)

  plot(exp1)

  clf_only_model <- create_model(train_x,train_y, NEURONS_CLF, DIMENSION_CLF)
  clf_only_model$to(device = dev)
  pretrain_net(N_EP_ONLY, clf_only_model, dsl, model_type = 1, LEARNING_RATE_CLF,
               sensitive_test)

  pre_clf_model <- create_model(train_x,train_y, NEURONS_CLF, DIMENSION_CLF)
  pre_clf_model$to(device = dev)
  pretrain_net(N_EP_PRECLF, pre_clf_model, dsl, model_type = 1, LEARNING_RATE_CLF,
               sensitive_test)

  prepared_data <- prepare_to_adv(p_preds[,2], sensitive_train, PARTITION)

  dsl_adv <- dataset_loader(prepared_data$train_x,prepared_data$train_y,
                            prepared_data$test_x,prepared_data$test_y,
                            batch_size = BATCH_SIZE)

  adv_model <- create_model(prepared_data$train_x,prepared_data$train_y,
                            neurons = NEURONS_ADV, dimensions = DIMENSION_ADV)

  adv_model$to(device = dev)
  pretrain_net(N_EP_PREADV, adv_model, dsl_adv, model_type = 0, LEARNING_RATE_ADV,
               sensitive_test)

  prepared_data$test_y

  make_preds(adv_model,dsl_adv$test_ds)

  eval_clf(adv_model,dsl_adv$test_ds)

  train_PAN(N_EP_PAN, dsl, clf_model, adv_model, dev, sensitive_train,
            sensitive_test, BATCH_SIZE, LEARNING_RATE_ADV,
            LEARNING_RATE_CLF, LAMBDA)

  exp <-  GAN_explainer(test_y,clf_model,DATA,PROTECTED,PRIVILIGED)

  plot(exp)
}
