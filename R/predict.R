#' Makes binary predictions
#'
#' These two functions make binary predictions of provided model based on test data set.
#'
#' @param model neural network classification model
#' @param test_ds dataset object used for making test predictions
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return integer(binary) vector of predictions
#' @export
#'
#' @examples
make_preds <- function(model,test_ds,dev){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont$to(device = dev))
  preds <- output$to(device = "cpu") %>% as.array()
  preds <- ifelse(preds[,1] < preds[,2], 2, 1)
  return(preds)

}

#' Makes probabilistic predictions
#'
#' These two functions make probabilistic predictions of provided model based on test
#' data set.
#'
#' @param model neural network classification model
#' @param test_ds dataset object used for making test predictions
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return float (probabilistic) vector of predictions
#' @export
#'
#' @examples
make_preds_prob <- function(model,test_ds,dev){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont$to(device = dev))
  preds <- output$to(device = "cpu") %>% as.array()
  return(preds)

}


