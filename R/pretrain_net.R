

#' Pretrains the neural network
#'
#' The function pretrain both the classifier and adversarial model. You can select which
#' model it is by setting model_type parameter
#'
#' @param n_epochs integer setting number of epochs for training. Default: 15
#' @param model neural network model we want to train
#' @param dsl dataset_loader object for the training
#' @param model_type indicates which model we train (0 for adversarial, 1 for classifier)
#' @param learning_rate float from [0,1] setting learning rate for model. Default: 0.001
#' @param sensitive_test test vector for sensitive variable used to calculate STP
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return NULL
#' @export
#'
#' @examples
pretrain_net <- function(n_epochs=15,model,dsl,model_type, learning_rate=0.001, sensitive_test, dev, verbose=TRUE, monitor=TRUE){

  optimizer <- optim_adam(model$parameters, lr = learning_rate)

  calc_loss <- function(output,batch){
    loss <- nnf_cross_entropy(output, batch)
    return(loss)
  }

  train_eval <- function(model,dsl,optimizer, dev){
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
      loss <- calc_loss(output,b$y$to(device = dev))
      valid_losses <- c(valid_losses, loss$item())
    })

    return(list("train_loss"=mean(train_losses), "test_loss"= mean(valid_losses)))
  }

  if(model_type == 0){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,dsl,optimizer, dev)
      if(monitor){
        acc<-eval_accuracy(model, dsl$test_ds, dev)
        verbose_cat(sprintf("Preadversary at epoch %d: training loss: %3.3f, validation: %3.3f, accuracy: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc),verbose=verbose)
      }else{
        verbose_cat(sprintf("Preadversary at epoch %d: training loss: %3.3f, validation: %3.3f\n", epoch, losses$train_loss, losses$test_loss),verbose=verbose)
      }
    }
  }
  if(model_type == 1){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,dsl,optimizer, dev)
      if(monitor){
        acc<-eval_accuracy(model, dsl$test_ds, dev)
        stp<-calc_STP(model,dsl$test_ds,sensitive_test,dev)
        verbose_cat(sprintf("Preclassifier at epoch %d: training loss: %3.3f, validation: %3.3f, accuracy: %3.3f, STPR: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc,stp),verbose=verbose)
      }else{
        verbose_cat(sprintf("Preclassifier at epoch %d: training loss: %3.3f, validation: %3.3f\n", epoch, losses$train_loss, losses$test_loss),verbose=verbose)
      }
    }
  }
  if(model_type == 2){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,dsl,optimizer, dev)
      if(monitor){
        acc<-eval_accuracy(model, dsl$test_ds, dev)
        stp<-calc_STP(model,dsl$test_ds,sensitive_test,dev)
        verbose_cat(sprintf("Classifier only at epoch %d: training loss: %3.3f, validation: %3.3f, accuracy: %3.3f, STPR: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc,stp),verbose=verbose)
      }else{
        verbose_cat(sprintf("Classifier only at epoch %d: training loss: %3.3f, validation: %3.3f\n", epoch, losses$train_loss, losses$test_loss),verbose=verbose)
      }
    }
  }
  return(list("train_loss"=losses$train_loss, "test_loss"= losses$test_loss))
}
