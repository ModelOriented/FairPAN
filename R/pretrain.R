#' Pretrains both classifier and adversarial
#'
#' This function can create and pretrain both classifier and adversarial. The user can
#' also provide the structure of both networks (their models). The classifier can be
#' pretrained, hovewer the adversarial cannot.
#'
#' @param clf_model optional value, provide the pretrain with your own classification
#'                  neural network. Default: NULL
#' @param adv_model optional value, provide the pretrain with your own adversarial
#'                  neural network. Default: NULL
#' @param trained  0 if the classificator is untrained, 1 if the classificator is already
#' pretrained. Default: 0
#' @param train_x numeric, scaled matrix of predictors used for training
#' @param train_y numeric, scaled vector of target used for training
#' @param sensitive_train integer, vector of sensitive values used for training
#' @param sensitive_test integer, vector of sensitive values used for testing
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
#' @param n_ep_preadv integer setting number of epochs for preadversarials training.
#'                    Default : 10
#' @param dsl dataset_loader object from pretrain
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return list of two obejcts: clf_model and adv_model which are pretrained neural
#'         networks.
#' @export
#'
#' @examples
pretrain <- function(clf_model=NULL,adv_model=NULL,trained=0,train_x=NULL,train_y=NULL,
                     sensitive_train,sensitive_test, batch_size=50,partition=0.7,
                     neurons_clf=c(32,32,32),neurons_adv=c(32,32,32),dimension_clf=2,
                     dimension_adv=2,learning_rate_clf=0.001,
                     learning_rate_adv=0.001,n_ep_preclf=5,n_ep_preadv=10,dsl,dev){

  if(is.null(clf_model)){
    clf_model <- create_model(train_x,train_y, neurons_clf, dimension_clf)
  }
  clf_model$to(device = dev)
  if(trained==0){
    pretrain_net(n_ep_preclf, clf_model, dsl, model_type = 1, learning_rate_clf,
                 sensitive_test, dev)
  }
  p_preds <- make_preds_prob(clf_model, dsl$train_ds, dev)

  prepared_data <- prepare_to_adv(p_preds[,2], sensitive_train, partition)

  dsl_adv <- dataset_loader(prepared_data$train_x,prepared_data$train_y,
                            prepared_data$test_x,prepared_data$test_y,
                            batch_size = batch_size, dev)

  if(is.null(adv_model)){
    adv_model <- create_model(prepared_data$train_x,prepared_data$train_y,
                              neurons = neurons_adv, dimensions = dimension_adv)
  }
  adv_model$to(device = dev)

  pretrain_net(n_ep_preadv, adv_model, dsl_adv, model_type = 0, learning_rate_adv,
               sensitive_test, dev)

  return(list("clf_model"=clf_model, "adv_model"=adv_model))
}
