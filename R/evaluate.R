#' Calculates accuracy
#'
#' This function evaluates the model by calculating its accuracy
#'
#' @param model neural network model we want to evaluate
#' @param test_ds \code{dataset} object from torch used for making test
#' predictions and evaluation
#' @param dev device used for calculations (cpu or gpu)
#'
#' @return double, accuracy of provided model
#' @export
#'
#' @examples
#'
#' \dontrun{
#' dev        <- if (torch::cuda_is_available())
#'                  torch_device("cuda:0") else "cpu"
#' # presaved torch model
#' model      <- torch_load("./tests/zzz/clf2")
#' # presaved output of `preprocess` function
#' processed  <- torch_load("./tests/zzz/processed")
#' dsl        <- dataset_loader(processed$train_x, processed$train_y,
#'                              processed$test_x, processed$test_y,
#'                              batch_size = 5, dev = dev)
#'
#' eval_accuracy(model, dsl$test_ds, dev)
#' }
#'
#' @import torch
#'
eval_accuracy <- function(model, test_ds, dev) {
  if (typeof(model) != 'closure')
    stop("provide a neural network as a model")
  if (typeof(test_ds) != "environment")
    stop("provide a test data set")
  if (typeof(test_ds$y) != "externalptr")
    stop("provide a test data set")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")
  model$eval()
  # We create a mega batch (with size of whole test_ds) and later take it into
  # evaluation
  test_dl     <- torch::dataloader(test_ds, batch_size = test_ds$.length(),
                            shuffle = FALSE)
  iter        <- test_dl$.iter()
  b           <- iter$.next()
  output      <- model(b$x_cont$to(device = dev))
  # We provide tensor of predictions into cpu to make regular operations on it
  preds       <- as.array(output$to(device = "cpu"))
  preds       <- ifelse(preds[, 1] < preds[, 2], 2, 1)
  comp_df     <- data.frame(preds = preds,
                            y = as.array(b$y$to(device = "cpu")))
  num_correct <- sum(comp_df$preds == comp_df$y)
  num_total   <- nrow(comp_df)
  accuracy    <- num_correct / num_total

  return(accuracy)
}

#' Calculates STP ratio
#'
#' Calculates Statistical Parity ((TP+FP)/(TP+FP+TN+FN)) ratio between
#' privileged and discriminated label for given model, which is the most
#' important fairness metric for us, because it measures how similar the
#' distributions of sensitive values are.
#'
#' @param model neural network model we want to evaluate
#' @param test_ds \code{dataset} object from torch used for making predictions
#' for STP ratio
#' @param sensitive numerical vector of sensitive variable
#' @param dev device used for calculations (cpu or gpu)
#'
#' @return float, STP ratio
#' @export
#'
#' @examples
#'
#' \dontrun{
#' dev        <- if (torch::cuda_is_available())
#'                 torch_device("cuda:0") else "cpu"
#'
#' # presaved torch model
#' model     <- torch_load("./tests/zzz/clf2")
#'
#' # presaved output of `preprocess` function
#' processed  <- torch_load("./tests/zzz/processed")
#'
#' dsl        <- dataset_loader(processed$train_x, processed$train_y,
#'                              processed$test_x, processed$test_y,
#'                              batch_size = 5, dev = dev)
#'
#' calc_STP(model, dsl$test_ds, processed$sensitive_test, dev)
#' }
#'
calc_STP <- function(model, test_ds, sensitive, dev) {

  if (typeof(model) != 'closure')
    stop("provide a neural network as a model")
  if (typeof(test_ds) != "environment")
    stop("provide a test data set")
  if (typeof(test_ds$y) != "externalptr")
    stop("provide a test data set")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")
  if (!is.vector(sensitive))
    stop("sensitive must be a vector")

  # We substract 1 from every prediction to get 0/1 labels
  preds <- make_preds(model, test_ds, dev) - 1
  real <- as.array((test_ds$y$to(device = "cpu"))) - 1
  sensitive <- sensitive - 1

  TP0<-0;FP0<-0;TN0<-0;FN0<-0;Tr0<-0;Fa0<-0
  TP1<-0;FP1<-0;TN1<-0;FN1<-0;Tr1<-0;Fa1<-0

  # We calculate TP and so on
  for (i in 1:length(preds)) {
    if (sensitive[i] == 0) {
      if (preds[i] == 1 & real[i] == 1) {
        TP0 = TP0 + 1
      } else if (preds[i] == 1 & real[i] == 0) {
        FP0 = FP0 + 1
      } else if (preds[i] == 0 & real[i] == 0) {
        TN0 = TN0 + 1
      } else if (preds[i] == 0 & real[i] == 1) {
        FN0 = FN0 + 1
      }
      if (real[i] == 1) {
        Tr0 = Tr0 + 1
      } else{
        Fa0 = Fa0 + 1
      }
    } else{
      if (preds[i] == 1 & real[i] == 1) {
        TP1 = TP1 + 1
      } else if (preds[i] == 1 & real[i] == 0) {
        FP1 = FP1 + 1
      } else if (preds[i] == 0 & real[i] == 0) {
        TN1 = TN1 + 1
      } else if (preds[i] == 0 & real[i] == 1) {
        FN1 = FN1 + 1
      }
      if (real[i] == 1) {
        Tr1 = Tr1 + 1
      } else{
        Fa1 = Fa1 + 1
      }
    }
  }
  # Calculation of STP ratio
  STP0 <- (TP0 + FP0) / (TP0 + FP0 + TN0 + FN0)
  STP1 <- (TP1 + FP1) / (TP1 + FP1 + TN1 + FN1)
  STPR <- min(c(STP0 / STP1, STP1 / STP0))

  return(STPR)
}
