#' Trains fairly the PAN model
#'
#' Trains Predictive Adversarial Network model, which means that it proceeds
#' with the mutual training of adversarial model on whole dataloader and
#' classifier on a single mini batch. The result is a fairer classifier.
#'
#' Trains Predictive Adversarial Network model. After the pretrain stage we
#' finally start proper fair training operation. We take all data from the
#' previous stage (models, optimizers and batch size) and engage the models
#' into a zero-sum-game where classifier starts to deceive adversarial and
#' in the end it ends with fair predictions. For a basic understanding of how
#' this process works it is advisable to read a vignette from this package. To
#' deeply understand this process you can read:
#' [Towards fairness in ML with adversarial networks, Stijn Tonk]
#' (https://godatadriven.com/blog/towards-fairness-in-ml-with-adversarial-networks/)
#'
#'
#' @param dsl \code{dataset_loader} object for classificator network.
#' @param clf_model net, nn_module, classifier model (preferably after pretrain).
#' @param adv_model net, nn_module, adversarial model (preferably after pretrain).
#' @param clf_optimizer optimizer for classificator model from pretrain.
#' @param adv_optimizer optimizer for adversarial model from pretrain.
#' @param dev device used for computation ("cuda" or "cpu").
#' @param sensitive_train integer vector of sensitive attribute used for
#' adversarial models training.
#' @param sensitive_test integer vector of sensitive attribute used for
#' adversarial models testing.
#' @param n_ep_pan number of epochs for PAN training. Default: 50.
#' @param batch_size batch size used in adversarial models \code{dataset_loader}
#' Default: 50.
#' @param learning_rate_adv learning rate of adversarial. Default: 0.001.
#' @param learning_rate_clf learning rate of classifier. Default: 0.001.
#' @param lambda parameter regulating learning process (intuition: the bigger it
#' is, the fairer predictions and the worse accuracy of classifier).
#' Default: 130
#' @param verbose logical indicating if we want to print monitored outputs or
#' not. Default: TRUE.
#' @param monitor logical indicating if we want to monitor the learning process
#' or not (monitoring tends to slow down the training proccess, but provides
#' some useful info to adjust parameters and training process) Default: TRUE.
#'
#' @return NULL if monitor is FALSE, list of metrics if it is TRUE
#' @export
#'
#' @examples
#' \dontrun{
#' dev <- "cpu"
#' # presaved torch models from pretrain phase
#' model1 <- torch_load(system.file("extdata","preclf",package="FairPAN"))
#' model11 <- torch_load(system.file("extdata","preadv",package="FairPAN"))
#'
#' # presaved optimizers dictionaries from pretrain phase
#' model1_optimizer_dict <- torch_load(system.file("extdata","preclf_optimizer",package="FairPAN"))
#' model11_optimizer_dict <- torch_load(system.file("extdata","preadv_optimizer",package="FairPAN"))
#'
#' # Recreating optimizers
#' model1_optimizer <- optim_adam(model1$parameters, lr = 0.001)
#' model11_optimizer <- optim_adam(model11$parameters, lr = 0.001)
#'
#' # Loading saved optimizer state
#' model1_optimizer$load_state_dict(model1_optimizer_dict)
#' model11_optimizer$load_state_dict(model11_optimizer_dict)
#'
#' # presaved output of `preprocess` function
#' processed <- torch_load(system.file("extdata","processed",package="FairPAN"))
#'
#' dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
#'                       processed$test_y, batch_size=5, dev=dev)
#'
#' fair_train(
#'   n_ep_pan = 2,
#'   dsl = dsl,
#'   clf_model = model1,
#'   adv_model = model11,
#'   clf_optimizer = model1_optimizer,
#'   adv_optimizer = model11_optimizer,
#'   dev = dev,
#'   sensitive_train = processed$sensitive_train,
#'   sensitive_test = processed$sensitive_test,
#'   batch_size = 5,
#'   learning_rate_adv = 0.001,
#'   learning_rate_clf = 0.001,
#'   lambda = 130,
#'   verbose = TRUE,
#'   monitor = TRUE
#' )
#' }


fair_train <- function(dsl,
                       clf_model,
                       adv_model,
                       clf_optimizer,
                       adv_optimizer,
                       dev,
                       sensitive_train,
                       sensitive_test,
                       n_ep_pan = 50,
                       batch_size = 50,
                       learning_rate_adv = 0.001,
                       learning_rate_clf = 0.001,
                       lambda = 130,
                       verbose = TRUE,
                       monitor = TRUE) {


  train_PAN(dsl,
           clf_model,
           adv_model,
           clf_optimizer,
           adv_optimizer,
           dev,
           sensitive_train,
           sensitive_test,
           n_ep_pan,
           batch_size,
           learning_rate_adv,
           learning_rate_clf,
           lambda,
           verbose,
           monitor)

}
