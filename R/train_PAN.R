

#' Trains PAN model
#'
#' Trains Predictive Adversarial Network model, which means that it proceeds with the
#' mutual training of adverssarial on whole data set and classifier on a single mini batch.
#' The result is a fairer classifier.
#'
#' @param N_EP_PAN number of epochs for PAN training
#' @param dsl dataset_loader object for classificator network
#' @param clf_model classifier model (preferably after pretrain)
#' @param adv_model adversarial model (preferably after pretrain)
#' @param dev device used to computation ("cuda" or "cpu")
#' @param sensitive_train integer vector of sensitive attribute used for training
#' @param sensitive_testnteger vector of sensitive attribute used for testing
#' @param BATCH_SIZE batch size used in adversarials dataset_loader
#' @param LEARNING_RATE_ADV learning rate of adversarial
#' @param LEARNING_RATE_CLF learning rate of classifier
#' @param LAMBDA parameter regulating learning proccess (intuition: the bigger it is,
#'               the fairer predictions).
#'
#' @return
#' @export
#'
#' @examples
train_PAN <- function(N_EP_PAN, dsl, clf_model, adv_model, dev, sensitive_train,
                      sensitive_test, BATCH_SIZE, LEARNING_RATE_ADV=,
                      LEARNING_RATE_CLF, LAMBDA){

  adversary_losses<-c()
  classifier_losses<-c()
  STP<-c()
  adversary_acc<-c()
  classifier_acc<-c()

  for (epoch in 1:N_EP_GAN){
    cat(sprintf("GAN epoch %d \n", epoch))
    train_dl <- dsl$train_ds %>% dataloader(batch_size = dsl$train_ds$.length(),
                                            shuffle = FALSE)
    iter <- train_dl$.iter()
    b <- iter$.next()
    output <- clf_model(b$x_cont$to(device = dev))
    preds<-output[,2]$to(device = "cpu")

    train_x <- as.numeric(preds)
    train_x <- matrix(train_x, ncol=1)
    train_y <- sensitive_train

    adv_dsl <- dataset_loader(train_x,train_y,train_x,train_y,BATCH_SIZE)

    adv_optimizer <- optim_adam(adv_model$parameters, lr = LEARNING_RATE_ADV)
    clf_optimizer <- optim_adam(clf_model$parameters, lr = LEARNING_RATE_CLF)
    adv_model$train()
    clf_model$train()
    train_losses <- c()
    clf_train_losses <- c()
    iterator<-dsl$train_dl$.iter()
    coro::loop(for (b in adv_dsl$train_dl) {
      adv_optimizer$zero_grad()
      output <- adv_model(b$x_cont$to(device = dev))
      loss <- nnf_cross_entropy(output, b$y$to(device = dev))*LAMBDA
      loss$backward()
      adv_optimizer$step()
      train_losses <- c(train_losses, loss$item())

    })

    iterator<-adv_dsl$train_dl$.iter()
    b <- iterator$.next()
    output <- adv_model(b$x_cont$to(device = dev))
    loss <- nnf_cross_entropy(output, b$y$to(device = dev))*LAMBDA

    iterator<-dsl$train_dl$.iter()
    b <- iterator$.next()
    clf_optimizer$zero_grad()
    clf_output <- clf_model(b$x_cont$to(device = dev))
    clf_loss <- nnf_cross_entropy(clf_output, b$y$to(device = dev))-loss$item()
    clf_loss$backward()
    clf_optimizer$step()


    if(epoch/1 == as.integer(epoch/1)){
      acc<-eval_clf(adv_model,adv_dsl$test_ds)
      adversary_acc<-c(adversary_acc,acc)
      cat(sprintf("Adversary accuracy at epoch %d: %3.3f\n", epoch,acc))

      cacc<-eval_clf(clf_model,dsl$test_ds)
      classifier_acc<-c(classifier_acc,cacc)
      cat(sprintf("Classifier accuracy at epoch %d: %3.3f\n", epoch,cacc))
    }
    stp<-STP_calc(clf_model,dsl$test_ds,sensitive_test)
    STP<-c(STP,stp)
    adversary_losses<-c(adversary_losses,mean(train_losses))
    #classifier_losses<-c(classifier_losses,clf_train_losses)
    cat(sprintf("Adversary Loss at epoch %d: training: %3.3f, STPR: %3.3f\n",
                epoch, mean(train_losses),stp))
    #cat(sprintf("Classifier Loss at epoch %d: training: %3.3f\n", epoch,
    #mean(clf_train_losses)))
  }
}
