

#' Create data sets and dataloaders
#'
#' Create two data sets, from given ..x matrices and ..y vectors and converts them into
#' dataloaders with provided batch size.
#'
#' @param train_x numerical matrix with training predictor variables
#' @param train_y numerical vector with training target variables
#' @param test_x numerical matrix with test predictor variables
#' @param test_y numerical vector with test target variables
#' @param batch_size integer defining the size of batch size in dataloader
#'
#' @return list of two data sets and two dataloaders for train and test respectively
#' @export
#'
#' @examples


dataset_loader <- function(train_x,train_y,test_x,test_y,batch_size=25){
  new_dataset <- dataset(

    name = "new_dataset",

    initialize = function(df,y2) {
      df <- na.omit(df)
      x_cont <- df
      self$x_cont <- torch_tensor(x_cont)$to(device = dev)
      self$y <- torch_tensor(y2,dtype = torch_long())$to(device = dev)
    },
    .getitem = function(i) {
      list(x_cont = self$x_cont[i, ], y=self$y[i])
    },
    .length = function() {
      self$y$size()[[1]]
    }
  )

  train_ds <- new_dataset(train_x,train_y)
  test_ds <- new_dataset(test_x,test_y)
  train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)
  test_dl <- test_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)

  return(list("train_ds" = train_ds,"test_ds"=test_ds,"train_dl"=train_dl,"test_dl"=test_dl))
}

#' Prepares data for adversarials pretrain
#'
#' Prepares classifiers output for adversarial by splitting original predictions into
#' train and test vectors
#'
#' @param preds numeric vector of predictions of target value made by classifier (preferably the probabilistic ones)
#' @param sensitive integer vector of sensitive attribute which adversarial has to predict
#' @param PARTITION float from [0,1] range meaning the size of train vector (test equals 1-PARTITION)
#'
#' @return list of four numeric lists with x and y data for train and test respectively
#' @export
#'
#' @examples
#'
#'
prepare_to_adv <- function(preds, sensitive, PARTITION){
  set.seed(123)
  train_indices <- sample(1:length(preds),  length(preds) * PARTITION)
  train_x <- as.numeric(preds[train_indices])
  train_x <- matrix(train_x, ncol=1)
  train_y <- sensitive[train_indices]
  test_x <- as.numeric(preds[setdiff(1:length(preds), train_indices)])
  test_x <- matrix(test_x, ncol=1)
  test_y <- sensitive[setdiff(1:length(sensitive), train_indices)]
  return(list("train_x"=train_x,"train_y"=train_y,"test_x"=test_x,"test_y"=test_y))
}
