#' Create data sets and dataloaders
#'
#' Create two data sets, from given ..x matrices and ..y vectors and converts them into
#' dataloaders with provided batch size.
#'
#' @param train_x numeric, scaled matrix of predictors used for training
#' @param test_x integer, matrix of predictors used for testing
#' @param train_y numeric, scaled vector of target used for training
#' @param test_y integer, vector of predictors used for testing
#' @param batch_size integer indicating a batch size used in dataloader. Default: 50
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return list of two data sets and two dataloaders for train and test respectively
#' @export
#'
#' @examples


dataset_loader <- function(train_x,train_y,test_x,test_y,batch_size=50,dev){
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
#' @param partition float from [0,1] range setting the size of train vector (test size
#'                  equals 1-partition). Default = 0.7.
#'
#' @return list of four numeric lists with x and y data for train and test respectively
#' @export
#'
#' @examples
#'
#'
prepare_to_adv <- function(preds, sensitive, partition=0.7){
  set.seed(123)
  train_indices <- sample(1:length(preds),  length(preds) * partition)
  train_x <- as.numeric(preds[train_indices])
  train_x <- matrix(train_x, ncol=1)
  train_y <- sensitive[train_indices]
  test_x <- as.numeric(preds[setdiff(1:length(preds), train_indices)])
  test_x <- matrix(test_x, ncol=1)
  test_y <- sensitive[setdiff(1:length(sensitive), train_indices)]
  return(list("train_x"=train_x,"train_y"=train_y,"test_x"=test_x,"test_y"=test_y))
}
