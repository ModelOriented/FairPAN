

#' Pretrains the neural network
#'
#'
#'
#'
#' @param n_epochs indicates the number of epochs for neural network pretrain
#' @param model neural network model we want to train
#' @param dsl dataset_loader object for the training
#' @param model_type indicates which model (0 for adversarial, 1 for classifier)
#' @param learnig_rate float, setting learning rate for the training
#' @param sensitive_test test vector for sensitive variable used to calculate STP
#'
#' @return
#' @export
#'
#' @examples
pretrain_net <- function(n_epochs=15,model,dsl,model_type, learnig_rate, sensitive_test){

  optimizer <- optim_adam(model$parameters, lr = learnig_rate)

  calc_loss <- function(output,batch){
    loss <- nnf_cross_entropy(output, batch)
    return(loss)
  }

  train_eval <- function(model,dsl,optimizer){
    model$train()
    train_losses <- c()
    coro::loop(for (b in dsl$train_dl) {
      optimizer$zero_grad()
      output <- model(b$x_cont$to(device = dev))
      loss <- calc_loss(output,b$y$to(device = dev))
      loss$backward()
      optimizer$step()
      train_losses <- c(train_losses, loss$item())
    })
    model$eval()
    valid_losses <- c()
    coro::loop(for (b in dsl$test_dl) {
      output <- model(b$x_cont$to(device = dev))
      loss <- calc_loss(lambda,output,b$y$to(device = dev))
      valid_losses <- c(valid_losses, loss$item())
    })

    return(list("train_loss"=mean(train_losses), "test_loss"= mean(valid_losses)))
  }

  if(model_type == 0){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,dsl,optimizer)
      acc<-eval_accuracy(model, dsl$test_ds)
      cat(sprintf("Preadversary Loss at epoch %d: training: %3.3f, validation: %3.3f, accuracy: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc))    }
  }
  if(model_type == 1){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,dsl,optimizer)
      acc<-eval_accuracy(model, dsl$test_ds)
      stp<-calc_STP(model,dsl$test_ds,sensitive_test)
      cat(sprintf("Preclassifier Loss at epoch %d: training: %3.3f, validation: %3.3f, accuracy: %3.3f, stp: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc,stp))    }
  }
  return(list("train_loss"=losses$train_loss, "test_loss"= losses$test_loss))
}
