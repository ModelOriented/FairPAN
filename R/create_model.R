create_model <- function(train_x,train_y,neurons,dimensions,p=0){
  net <- nn_module(
    "net",
    initialize = function(n_cont, Neurons, output_dim, p) {
      torch_manual_seed(7)
      self$fc1<-nn_linear(n_cont, Neurons[1])
      for (i in 2:length(Neurons)){
        str<-paste("self$fc",i," <- nn_linear(Neurons[",i-1,"]", ",Neurons[",i,"])",sep="")
        eval(parse(text=str))
      }
      self$output <- nn_linear(Neurons[length(Neurons)], output_dim)

    },
    forward = function(x_cont) {
      all <- torch_cat(x_cont,dim=2)
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
    output_dim = length(levels(factor(train_y))),
    p=p
  )

}
