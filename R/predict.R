#' Makes binary/probabilistic predictions
#'
#' These two functions make binary or probabilistic predctions of provided model based on
#' test data set
#'
#' @param model neural network classification model
#' @param test_ds dataset object used for making test predictions
#'
#' @return integer(binary) or float(probabilistic) vector of predictions
#' @export
#'
#' @examples
make_preds <- function(model,test_ds){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont$to(device = dev))
  preds <- output$to(device = "cpu") %>% as.array()
  preds <- ifelse(preds[,1] < preds[,2], 2, 1)
  return(preds)

}

make_preds_prob <- function(model,test_ds){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont$to(device = dev))
  preds <- output$to(device = "cpu") %>% as.array()
  return(preds)

}


