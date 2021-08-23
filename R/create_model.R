#' Creates a simple neural network model
#'
#' Provides a method to create a simple neural network model which should be
#' enough for tabular data classification tasks. The model consists of `nn_linear`
#' layers, there are no dropouts and the activation function between the layers
#' is `nnf_relu`, whereas the last one is `nnf_softmax`. The user can provide
#' demanded architecture of the layers and select a softmaxes dimension.
#'
#' @param train_x numeric, scaled matrix of predictors used for training. Here
#' it is used for getting its size to build suitable neural network.
#' @param train_y numeric, scaled vector of target used for training Here
#' it is used for getting its size to build suitable neural network.
#' @param neurons numeric, vector of integers describing the architecture.
#' Notation c(8,16,8) means 3 layer neural network with 8,16 and 8 neurons in
#' 1st, 2nd and 3rd layer. Default: c(32,32,32)
#' @param dimensions integer 0,1 or 2 setting nnf_softmax dimension for
#' classifier. Default: 2 (suggested to use 2 for classifier and 1 for
#' adversarial)
#' @param seed integer, seed for initial weights, set NULL for none. Default: 7.
#'
#' @return net,nn_module, neural network model
#' @export
#'
#' @examples
#' train_x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
#' train_y <- c(1, 2, 3)
#' model   <- create_model(train_x,
#'                         train_y,
#'                         neurons = c(16, 8, 16),
#'                         dimensions = 1,
#'                         seed=7)
#' @import magrittr
#'

create_model <- function(train_x, train_y, neurons=c(32,32,32), dimensions=2, seed=7){

  if (!dimensions %in% c(0, 1, 2))
    stop("dimensions must be a 0,1 or 2")
  if (!is.vector(train_y))
    stop("train_y must be a vector")
  if (!is.matrix(train_x))
    stop("train_x must be a matrix")
  if (nrow(train_x) != length(train_y))
    stop("length of train_y must be equal number of rows of train_x")
  if (sum(neurons - neurons / 1) != 0)
    stop("neurons must be a vector of integers")

  #Without this NA self inside nn_module produces global variable note
  self <- NA

  net <- torch::nn_module(
    "net",
    initialize = function(n_cont, Neurons, output_dim) {
      # We're setting seed to have the same initial weights
      if(!is.null(seed)){
        torch::torch_manual_seed(seed)
      }
      self$fc1 <- torch::nn_linear(n_cont, Neurons[1])
      # We automatically create next layers of the network
      for (i in seq_len(length(Neurons)-1)+1) {
        str<-paste("self$fc",i," <- torch::nn_linear(Neurons[",i-1,"]",
                   ",Neurons[",i,"])",sep="")
        eval(parse(text = str))
      }
      self$output <- torch::nn_linear(Neurons[length(Neurons)], output_dim)

    },
    forward = function(x_cont) {
      # We concatenate the given batch of tensors
      all <- torch::torch_cat(x_cont, dim = dimensions)
      # Then we add Relu activation for inner layers
      for (i in seq_len(length(neurons))) {
        str<-paste("all<-all %>% self$fc",i,"() %>% torch::nnf_relu()",sep="")
        eval(parse(text = str))
      }
      # And softmax for the last one
      all %>% self$output() %>% torch::nnf_softmax(dim = dimensions)

    }
  )
  # Here we use train_x and train_y for getting our dimensions
  model <- net(
    n_cont = ncol(data.frame(train_x)),
    Neurons = neurons,
    output_dim = length(levels(factor(train_y)))
  )

  return(model)

}
