#' Create datasets and dataloaders
#'
#' Create two torch datasets, from given train_x, test_x matrices and train_y,
#' test_y vectors and converts them into torch dataloaders with provided batch
#' size.
#'
#' @param train_x numeric, scaled matrix of predictors used for training
#' @param test_x integer, matrix of predictors used for testing
#' @param train_y numeric, scaled vector of target used for training
#' @param test_y integer, vector of predictors used for testing
#' @param batch_size integer indicating a batch size used in dataloader.
#' Default: 50
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return list of two data sets and two dataloaders for train and test
#' respectively
#' @export
#'
#' @examples
#' train_x = matrix(c(1,2,3,4,5,6),nrow=3)
#' train_y = c(1,2,3)
#' test_x = matrix(c(1,2,3,4),nrow=2)
#' test_y = c(1,2)
#' dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"
#' dataset_loader(train_x,train_y,test_x,test_y,batch_size=1,dev)


dataset_loader <- function(train_x,
                           train_y,
                           test_x,
                           test_y,
                           batch_size = 50,
                           dev) {


  if (!is.numeric(train_x))
    stop("train_x must be numeric")
  if (!is.numeric(test_x))
    stop("test_x must be numeric")
  if (!is.numeric(train_y) ||
      !is.vector(train_y))
    stop("train_y must be numeric vector of target")
  if (!is.numeric(test_y) ||
      !is.vector(test_y))
    stop("test_y must be numeric vector of target")
  if (!is.numeric(batch_size))
    stop("batch size must be numeric")

  new_dataset <- dataset(

    name = "new_dataset",

    initialize = function(df,y2) {
      df <- na.omit(df)
      x_cont <- df
      #create tensors for x and y and pass it to device
      self$x_cont <- torch_tensor(x_cont)$to(device = dev)
      self$y <- torch_tensor(y2, dtype = torch_long())$to(device = dev)
    },
    .getitem = function(i) {
      list(x_cont = self$x_cont[i, ], y=self$y[i])
    },
    .length = function() {
      self$y$size()[[1]]
    }
  )

  #create datasets and data loaders
  train_ds <- new_dataset(train_x, train_y)
  test_ds  <- new_dataset(test_x, test_y)
  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = FALSE)
  test_dl  <- dataloader(test_ds, batch_size = batch_size, shuffle = FALSE)

  return(list("train_ds" = train_ds,"test_ds"=test_ds,
              "train_dl"=train_dl,"test_dl"=test_dl))
}

#' Prepares data for adversarial
#'
#' Prepares classifiers output for adversarial by splitting original predictions
#' into train and test vectors
#'
#' @param preds numeric vector of predictions of target value made by
#' classifier (preferably the probabilistic ones)
#' @param sensitive integer vector of sensitive attribute which adversarial has
#' to predict
#' @param partition float from [0,1] range setting the size of train vector
#' (test size equals 1-partition). Default = 0.7.
#'
#' @return list of four numeric lists with x and y data for train and test
#' respectively
#' @export
#'
#' @examples
#'
#' preds <-c(0.312,0.343,0.932,0.754,0.436,0.185,0.527,0.492,0.743,0.011)
#' sensitive <- c(1,1,2,2,1,1,2,2,2,1)
#' prepare_to_adv(preds,sensitive,partition=0.6)
#'
prepare_to_adv <- function(preds, sensitive, partition=0.7){
  if (!is.numeric(preds) || !is.vector(preds))
    stop("preds must be numeric vector of probabilities")
  if (!is.numeric(sensitive) || !is.vector(sensitive))
    stop("sensitive must be numeric vector of mapped sensitive classes")
  if (!is.numeric(partition) || partition > 1 || partition < 0)
    stop("partition must be numeric be in [0,1]")

  set.seed(123)
  train_indices <- sample(1:length(preds),  length(preds) * partition)
  train_x       <- as.numeric(preds[train_indices])
  train_x       <- matrix(train_x, ncol = 1)
  train_y       <- sensitive[train_indices]
  test_x        <- as.numeric(preds[setdiff(1:length(preds), train_indices)])
  test_x        <- matrix(test_x, ncol = 1)
  test_y        <- sensitive[setdiff(1:length(sensitive), train_indices)]
  return(list("train_x"=train_x,"train_y"=train_y,
              "test_x"=test_x,"test_y"=test_y))
}


#' Preprocesses data for training
#'
#' This function prepares provided dataset to be ready for the training process.
#' It  makes data suitable for training functions, splits it into train, test
#' and validation, provides other data objects that are necessary for our
#' training.
#'
#' @param data list representing whole table of data.
#' @param target_name character, column name of the target variable. Selected
#' column must be interpretable as categorical.
#' @param sensitive_name character, column name of the sensitive variable.
#' Selected column must be interpretable as categorical.
#' @param privileged character meaning the name of privileged group
#' @param discriminated character meaning the name of discriminated group
#' @param drop_also character vector, column names of other columns to drop
#' (like other sensitive variables).
#' @param sample double from [0,1] setting size of our sample from original
#' data set. Default: 1
#' @param train_size double from [0,1] setting size of our train. Note that
#' train_size+test_size+validation_size=1. Default=0.7
#' @param test_size double from [0,1] setting size of our test Note that
#' train_size+test_size+validation_size=1. Default=0.3
#' @param validation_size double from [0,1] setting size of our validation.
#' Note that train_size+test_size+validation_size=1. Default=0
#' @param seed sets seed for the sampling for code reproduction. Default=NULL
#'
#' @return list of prepared data
#' (
#' train_x, - numeric scaled matrix for classifier training
#' train_y, - numeric scaled vector for classifier training
#' sensitive_train, - numeric scaled vector for adversaries training
#' test_x, - numeric scaled matrix for classifier testing
#' test_y, - numeric scaled vector for classifier testing
#' sensitive_test, - numeric scaled vector for adversaries testing
#' valid_x, - numeric scaled matrix for classifier validation
#' valid_y, - numeric scaled vector for classifier validation
#' sensitive_valid, - numeric scaled vector for adversaries validation
#' data_scaled_test, - numeric scaled data set for testing
#' data_scaled_valid, - numeric scaled data set for validation
#' data_test, - whole dataset for testing, unchanged
#' protected_test, - character vector of protected values for explainers test
#' data_valid, - whole dataset for validation, unchanged
#' protected_valid - character vector of protected values for explainers valid
#' )
#' @export
#'
#' @examples
#' data("adult")
#'
#' processed <-
#'   preprocess(
#'     adult,
#'     "salary",
#'     "sex",
#'     "Male",
#'     "Female",
#'     c("race"),
#'     sample = 0.05,
#'     train_size = 0.65,
#'     test_size = 0.35,
#'     validation_size = 0,
#'     seed = 7
#'   )
#'
preprocess <- function(data,
                       target_name,
                       sensitive_name,
                       privileged,
                       discriminated,
                       drop_also,
                       sample = 1,
                       train_size = 0.7,
                       test_size = 0.3,
                       validation_size = 0,
                       seed = NULL) {



  if (train_size + test_size + validation_size != 1)
    stop("train_size+test_size+validation_size must equal 1")
  if (!is.character(target_name) || !is.character(sensitive_name))
    stop("target_name and sensitive_name must be characters")
  if (!is.character(drop_also))
    stop("drop_also must be a character vector")
  if (!is.list(data))
    stop("data must be a list")

  col    <- eval(parse(text = paste("data$", sensitive_name, sep = "")))
  #balance dataset to have the same number of sensitive values, so
  #adversarial doesn't overfit (like all predictions are 1 or 2)
  M      <- min(table(col))
  df_new <- data[col == privileged,][1:M,]
  df_new <- rbind(df_new, data[col == discriminated,][1:M,])
  data   <- df_new
  data   <- na.omit(data)

  set.seed(seed)

  sample_indices <- sample(1:nrow(data), nrow(data) * sample)
  data           <- data[sample_indices, ]
  data           <- na.omit(data)

  sensitive <-
    as.integer (eval(parse(text = paste(
      "data$", sensitive_name, sep = ""
    ))))

  target <-
    as.integer (eval(parse(text = paste(
      "data$", target_name, sep = ""
    ))))

  #drop columns we dont want to be in learning set
  data_coded <- data[, -which(names(data) %in%
                                c(target_name, sensitive_name, drop_also))]
  #encode columns which are not numeric
  for (i in 1:ncol(data_coded)) {
    if (!is.numeric(data_coded[, i])) {
      data_coded[, i] <- as.integer(data_coded[, i])
    }
  }

  #prepare data with scaling
  data_matrix <- matrix(unlist(data_coded), ncol = ncol(data_coded))
  data_scaled <- scale(data_matrix, center = TRUE, scale = TRUE)
  #prepare indices for all classes
  set.seed(seed)
  train_indices <- sample(1:nrow(data_coded), train_size * nrow(data_coded))
  rest_indices  <- setdiff(1:nrow(data_coded), train_indices)
  set.seed(seed)
  test_indices       <- sample(rest_indices, test_size /
                            (1 - train_size) * length(rest_indices))
  validation_indices <- setdiff(rest_indices, test_indices)

  data_scaled_test   <- data_scaled[test_indices, ]
  data_scaled_valid  <- data_scaled[validation_indices, ]

  train_x            <- data_scaled[train_indices, ]
  train_y            <- target[train_indices]
  sensitive_train    <- sensitive[train_indices]

  test_x             <- data_scaled[test_indices, ]
  test_y             <- target[test_indices]
  sensitive_test     <- sensitive[test_indices]

  valid_x            <- data_scaled[validation_indices,]
  valid_y            <- target[validation_indices]
  sensitive_valid    <- sensitive[validation_indices]


  data_test          <- data[test_indices, ]
  protected_test <-
    eval(parse(text = paste("data_test$", sensitive_name, sep = "")))

  data_valid         <- data[validation_indices, ]
  protected_valid <-
    eval(parse(text = paste("data_valid$", sensitive_name, sep = "")))

  prepared_data <- list(
    "train_x" = train_x,
    "train_y" = train_y,
    "sensitive_train" = sensitive_train,
    "test_x" = test_x,
    "test_y" = test_y,
    "sensitive_test" = sensitive_test,
    "valid_x" = valid_x,
    "valid_y" = valid_y,
    "sensitive_valid" = sensitive_valid,
    "data_scaled_test" = data_scaled_test,
    "data_scaled_valid" = data_scaled_valid,
    "data_test" = data_test,
    "protected_test" = protected_test,
    "data_valid" = data_valid,
    "protected_valid" = protected_valid
  )

  return(prepared_data)
}
