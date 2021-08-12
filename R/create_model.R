#' Creates a default neural network model
#'
#' Provides a method to create a simple neural network model which should be enough for
#' tabular data classification tasks. The model consists of nn_linear layers, there are
#' no dropouts and the activation function between the layers is nnf_relu(), whereas the
#' last one is nnf_softmax. The user can provide demanded architecture of the leyers and
#' select a softmaxes dimension.
#'
#' @param train_x numeric, scaled matrix of predictors used for training
#' @param train_y numeric, scaled vector of target used for training
#' @param neurons vector of integers describing the architecture. Notation c(8,16,8) means
#'                3 layer neural network with 8,16 and 8 neurons in 1st, 2nd and 3rd
#'                layer. Default: c(32,32,32)
#' @param dimensions integer from [0,2] setting nnf_softmax dimension for classifier.
#'                   Default: 2 (suggested to use 2 for classifier and 1 for adversarial)
#'
#' @return neural network model
#'
#' @examples
create_model <- function(train_x,train_y,neurons=c(32,32,32),dimensions=2){
  if(!dimensions %in% c(0,1,2)) stop("dimensions must be a 0,1 or 2")
  if(!is.vector(train_y)) stop("train_y must be a vector")
  if(!is.matrix(train_x)) stop("train_x must be a matrix")
  if(nrow(train_x)!=length(train_y)) stop("length of train_y must be equal number of rows of train_x")
  if(sum(neurons-neurons/1)!=0) stop("neurons must be a vector of integers")

  net <- nn_module(
    "net",
    initialize = function(n_cont, Neurons, output_dim) {
      torch_manual_seed(7)
      self$fc1<-nn_linear(n_cont, Neurons[1])
      for (i in 2:length(Neurons)){
        str<-paste("self$fc",i," <- nn_linear(Neurons[",i-1,"]", ",Neurons[",i,"])",sep="")
        eval(parse(text=str))
      }
      self$output <- nn_linear(Neurons[length(Neurons)], output_dim)

    },
    forward = function(x_cont) {
      all <- torch_cat(x_cont,dim=dimensions)
      for (i in 1:length(neurons)){
        str<-paste("all<-all %>% self$fc",i,"() %>%
                    nnf_relu()",sep="")
        eval(parse(text=str))
      }
      all %>% self$output() %>% nnf_softmax(dim = dimensions)

    }
  )
  model <- net(
    n_cont = ncol(data.frame(train_x)),
    Neurons = neurons,
    output_dim = length(levels(factor(train_y)))
  )

  return(model)

}
