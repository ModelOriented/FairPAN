#' Pretrains both classifier and adversarial
#'
#' This function can create and pretrain both classifier and adversarial. The user can
#' also provide the structure of both networks (their models). The classifier can be
#' pretrained, hovewer the adversarial cannot.
#'
#' @param clf_model optional value, provide the pretrain with your own classification
#' neural network. Default: NULL
#' @param adv_model optional value, provide the pretrain with your own adversarial
#' neural network. Default: NULL
#' @param clf_optimizer optional value, provide the optimizer of classifier if you decided
#' to provide your own pre trained classifier. Default: NULL
#' @param trained  0 if the classificator is untrained, 1 if the classificator is already
#' pretrained. Default: 0
#' @param train_x numeric, scaled matrix of predictors used for training
#' @param train_y numeric, scaled vector of target used for training
#' @param sensitive_train integer, vector of sensitive values used for training
#' @param sensitive_test integer, vector of sensitive values used for testing
#' @param batch_size integer indicating a batch size used in dataloader. Default: 50
#' @param partition float from [0,1] range setting the size of train vector (test size
#' equals 1-partition). Default = 0.7.
#' @param neurons_clf integer vector describing a neural architecture of classifier
#' network. Default: c(32,32,32). This notation means that the network has 3 layers with
#' 32 neurons each.
#' @param neurons_adv integer vector describing a neural architecture of adversarial
#' network. Default: c(32,32,32). This notation means that the network has 3 layers with
#' 32 neurons each.
#' @param dimension_clf integer from [0,2] setting nnf_softmax dimension for classifier.
#' Default: 2 (suggested to use 2 for classifier and 1 for adversarial)
#' @param dimension_adv integer from [0,2] setting nnf_softmax dimension for adversarial.
#' Default: 1 (suggested to use 2 for classifier and 1 for adversarial)
#' @param learning_rate_clf float from [0,1] setting learning rate for classifier.
#' Default: 0.001
#' @param learning_rate_adv float from [0,1] setting learning rate for classifier.
#' Default: 0.001
#' @param n_ep_preclf integer setting number of epochs for preclassifiers training.
#' Default: 5
#' @param n_ep_preadv integer setting number of epochs for preadversarials training.
#' Default : 10
#' @param dsl dataset_loader object from pretrain
#' @param dev device used to calculations (cpu or gpu)
#' @param verbose logical indicating if we want to print monitored outputs or not
#' @param monitor logical indicating if we want to monitor the learning process or not
#' (monitoring tends to slow down the training proccess, but provides some useful info to
#' adjust parameters and training process)
#'
#' @return list of two obejcts: clf_model and adv_model which are pretrained neural
#' networks.
#' @export
#'
#' @examples
pretrain <- function(clf_model=NULL,adv_model=NULL,clf_optimizer=NULL,trained=FALSE,train_x=NULL,train_y=NULL,
                     sensitive_train,sensitive_test, batch_size=50,partition=0.7,
                     neurons_clf=c(32,32,32),neurons_adv=c(32,32,32),dimension_clf=2,
                     dimension_adv=1,learning_rate_clf=0.001,
                     learning_rate_adv=0.001,n_ep_preclf=5,n_ep_preadv=10,dsl,dev,verbose=TRUE,monitor=TRUE){

  #add ifs for optimizer
  if(n_ep_preclf!=n_ep_preclf/1 || n_ep_preclf<0) stop("n_ep_preclf must be a positive integer")
  if(n_ep_preadv!=n_ep_preadv/1 || n_ep_preadv<0) stop("n_ep_preadv must be a positive integer")
  if(typeof(dsl)!="list") stop("dsl must be list of 2 data sets and 2 data loaders from dataset_loader function")
  if(typeof(dsl$test_ds)!="environment") stop("dsl must be list of 2 data sets and 2 data loaders from dataset_loader function")
  if(typeof(dsl$test_ds$y)!="externalptr") stop("dsl must be list of 2 data sets and 2 data loaders from dataset_loader function")
  if(learning_rate_clf>1 || learning_rate_clf<0) stop("learning_rate_clf must be between 0 and 1")
  if(learning_rate_adv>1 || learning_rate_adv<0) stop("learning_rate_adv must be between 0 and 1")
  if(partition>1 || partition<0) stop("partition must be between 0 and 1")

  if(!dev %in% c("gpu","cpu"))stop("dev must be gpu or cpu")
  if(!is.vector(sensitive_test)) stop("sensitive_test must be a vector")
  if(!is.vector(sensitive_train)) stop("sensitive_train must be a vector")

  if(!is.logical(trained))stop("trained must be logical")
  if(!is.logical(verbose)||!is.logical(monitor)) stop("verbose and monitor must be logical")

  if(!dimension_clf %in% c(0,1,2)) stop("dimension_clf must be a 0,1 or 2")
  if(!dimension_adv %in% c(0,1,2)) stop("dimension_adv must be a 0,1 or 2")
  if(sum(neurons_clf-neurons_clf/1)!=0) stop("neurons_clf must be a vector of integers")
  if(sum(neurons_adv-neurons_adv/1)!=0) stop("neurons_adv must be a vector of integers")

  if(!is.null(clf_optimizer)&&!typeof(clf_optimizer)=="environment") stop("clf_optimizer mus be NULL or optimizer from clf pretrain")


  if(is.null(clf_model)){
    if(!is.vector(train_y)) stop("train_y must be a vector")
    if(!is.matrix(train_x)) stop("train_x must be a matrix")
    if(nrow(train_x)!=length(train_y)) stop("length of train_y must be equal number of rows of train_x")
    clf_model <- create_model(train_x,train_y, neurons_clf, dimension_clf)
  }
  if(typeof(clf_model)!='closure') stop("provide a neural network as a model")
  clf_model$to(device = dev)
  if(!trained){
    clf_optimizer<-pretrain_net(n_ep_preclf, clf_model, dsl, model_type = 1, learning_rate_clf,
                 sensitive_test, dev,verbose=verbose,monitor = monitor)
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
  if(typeof(adv_model)!='closure') stop("provide a neural network as a model")
  adv_model$to(device = dev)

  adv_optimizer <- pretrain_net(n_ep_preadv, adv_model, dsl_adv, model_type = 0, learning_rate_adv,
               sensitive_test, dev,verbose=verbose,monitor = monitor)

  return(list("clf_model"=clf_model, "adv_model"=adv_model, "clf_optimizer"=clf_optimizer,"adv_optimizer"=adv_optimizer))
}
