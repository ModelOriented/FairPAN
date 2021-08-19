#' Makes binary predictions
#'
#' Makes binary predictions for classification model
#'
#' @param model neural network classification model
#' @param test_ds \code{dataset} object from torch used for making test
#' predictions
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return integer (binary) vector of predictions
#' @export
#'
#' @examples
#'
#' \dontrun{
#' dev       <- if (torch::cuda_is_available())
#'                torch_device("cuda:0") else "cpu"
#'
#' model1    <- torch_load("./tests/zzz/preclf")
#' processed <- torch_load("./tests/zzz/processed")
#'
#' dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
#'                       processed$test_y, batch_size=5, dev=dev)
#'
#' preds1   <- make_preds(model1,dsl$test_ds,dev)
#' }


make_preds <- function(model, test_ds, dev) {

  if (typeof(model) != 'closure')
    stop("provide a neural network as a model")
  if (typeof(test_ds) != "environment")
    stop("provide a test data set")
  if (typeof(test_ds$y) != "externalptr")
    stop("provide a test data set")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")

  model$eval()

  test_dl <- torch::dataloader(test_ds, batch_size = test_ds$.length(),
                        shuffle = FALSE)
  iter    <- test_dl$.iter()
  b       <- iter$.next()
  output  <- model(b$x_cont$to(device = dev))
  preds   <- as.array(output$to(device = "cpu"))
  preds   <- ifelse(preds[, 1] < preds[, 2], 2, 1)

  return(preds)

}

#' Makes probabilistic predictions
#'
#' Makes probabilistic predictions for classification model
#'
#' @param model neural network classification model
#' @param test_ds \code{dataset} object from torch used for making test
#' predictions
#' @param dev device used for calculations (cpu or gpu)
#'
#' @return float (probabilistic) vector of predictions
#' @export
#'
#' @examples
#' \dontrun{
#' dev       <- if (torch::cuda_is_available())
#'                torch_device("cuda:0") else "cpu"
#'
#' model1    <- torch_load("./zzz/preclf")
#' processed <- torch_load("./zzz/processed")
#'
#' dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
#'                       processed$test_y, batch_size=5, dev=dev)
#'
#' preds1   <- make_preds_prob(model1,dsl$test_ds,dev)
#' }
#'
make_preds_prob <- function(model, test_ds, dev) {

  if (typeof(model) != 'closure')
    stop("provide a neural network as a model")
  if (typeof(test_ds) != "environment")
    stop("provide a test data set")
  if (typeof(test_ds$y) != "externalptr")
    stop("provide a test data set")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")

  model$eval()

  test_dl <- torch::dataloader(test_ds, batch_size = test_ds$.length(),
                        shuffle = FALSE)
  iter    <- test_dl$.iter()
  b       <- iter$.next()
  output  <- model(b$x_cont$to(device = dev))
  preds   <- as.array(output$to(device = "cpu"))

  return(preds)

}


