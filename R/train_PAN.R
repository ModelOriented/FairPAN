

#' Trains PAN model
#'
#' Trains Predictive Adversarial Network model, which means that it proceeds with the
#' mutual training of adverssarial on whole data set and classifier on a single mini batch.
#' The result is a fairer classifier.
#'
#' @param n_ep_pan number of epochs for PAN training
#' @param dsl dataset_loader object for classificator network
#' @param clf_model classifier model (preferably after pretrain)
#' @param adv_model adversarial model (preferably after pretrain)
#' @param dev device used to computation ("cuda" or "cpu")
#' @param sensitive_train integer vector of sensitive attribute used for training
#' @param sensitive_test integer vector of sensitive attribute used for testing
#' @param batch_size batch size used in adversarials dataset_loader
#' @param learning_rate_adv learning rate of adversarial
#' @param learning_rate_clf learning rate of classifier
#' @param lambda parameter regulating learning proccess (intuition: the bigger it is,
#'               the fairer predictions).
#'
#' @return NULL
#' @export
#'
#' @examples
train_PAN <- function(n_ep_pan, dsl, clf_model, adv_model, dev, sensitive_train,
                      sensitive_test, batch_size, learning_rate_adv,
                      learning_rate_clf, lambda, verbose, monitor){
  if(monitor){
    adversary_losses<-c()
    STP<-c()
    adversary_acc<-c()
    classifier_acc<-c()
  }

  for (epoch in 1:n_ep_pan){
    verbose_cat(sprintf("PAN epoch %d \n", epoch),verbose)
    train_dl <- dsl$train_ds %>% dataloader(batch_size = dsl$train_ds$.length(),
                                            shuffle = FALSE)
    iter <- train_dl$.iter()
    b <- iter$.next()
    output <- clf_model(b$x_cont$to(device = dev))
    preds<-output[,2]$to(device = "cpu")

    train_x <- as.numeric(preds)
    train_x <- matrix(train_x, ncol=1)
    train_y <- sensitive_train

    adv_dsl <- dataset_loader(train_x,train_y,train_x,train_y,batch_size,dev)

    adv_optimizer <- optim_adam(adv_model$parameters, lr = learning_rate_adv)
    clf_optimizer <- optim_adam(clf_model$parameters, lr = learning_rate_clf)
    adv_model$train()
    clf_model$train()
    train_losses <- c()
    clf_train_losses <- c()
    iterator<-dsl$train_dl$.iter()
    coro::loop(for (b in adv_dsl$train_dl) {
      adv_optimizer$zero_grad()
      output <- adv_model(b$x_cont$to(device = dev))
      loss <- nnf_cross_entropy(output, b$y$to(device = dev))*lambda
      loss$backward()
      adv_optimizer$step()
      train_losses <- c(train_losses, loss$item())

    })

    iterator<-adv_dsl$train_dl$.iter()
    b <- iterator$.next()
    output <- adv_model(b$x_cont$to(device = dev))
    loss <- nnf_cross_entropy(output, b$y$to(device = dev))*lambda

    iterator<-dsl$train_dl$.iter()
    b <- iterator$.next()
    clf_optimizer$zero_grad()
    clf_output <- clf_model(b$x_cont$to(device = dev))
    clf_loss <- nnf_cross_entropy(clf_output, b$y$to(device = dev))-loss$item()
    clf_loss$backward()
    clf_optimizer$step()

    adversary_losses<-c(adversary_losses,mean(train_losses))

    if(monitor){
      acc<-eval_accuracy(adv_model,adv_dsl$test_ds,dev)
      adversary_acc<-c(adversary_acc,acc)

      cacc<-eval_accuracy(clf_model,dsl$test_ds,dev)
      classifier_acc<-c(classifier_acc,cacc)

      stp<-calc_STP(clf_model,dsl$test_ds,sensitive_test,dev)
      STP<-c(STP,stp)

      verbose_cat(sprintf("Classifier accuracy at epoch %d: %3.3f\n", epoch,cacc),verbose)
      verbose_cat(sprintf("Adversary at epoch %d: training loss: %3.3f, accuracy: %3.3f, STPR: %3.3f\n",
                  epoch, mean(train_losses),acc,stp), verbose)
    }else{
      verbose_cat(sprintf("Adversary at epoch %d: training loss: %3.3f",
                          epoch, mean(train_losses)), verbose)
    }

  }
}
