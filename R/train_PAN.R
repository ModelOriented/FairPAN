#' Trains PAN model
#'
#' Trains Predictive Adversarial Network model, which means that it proceeds
#' with the mutual training of adversarial model on whole dataloader and
#' classifier on a single mini batch. The result is a fairer classifier.
#'
#' @param n_ep_pan number of epochs for PAN training
#' @param dsl \code{dataset_loader} object for classificator network
#' @param clf_model classifier model (preferably after pretrain)
#' @param adv_model adversarial model (preferably after pretrain)
#' @param clf_optimizer optimizer for classificator model from pretrain
#' @param adv_optimizer optimizer for adversarial model from pretrain
#' @param dev device used to computation ("cuda" or "cpu")
#' @param sensitive_train integer vector of sensitive attribute used for
#' training
#' @param sensitive_test integer vector of sensitive attribute used for testing
#' @param batch_size batch size used in adversarial models \code{dataset_loader}
#' @param learning_rate_adv learning rate of adversarial
#' @param learning_rate_clf learning rate of classifier
#' @param lambda parameter regulating learning process (intuition: the bigger it
#' is, the fairer predictions and the worse accuracy of classifier).
#' @param verbose logical indicating if we want to print monitored outputs or
#' not
#' @param monitor logical indicating if we want to monitor the learning process
#' or not (monitoring tends to slow down the training proccess, but provides
#' some useful info to adjust parameters and training process)
#'
#' @return NULL if monitor is FALSE, list of metrics if it is TRUE
train_PAN <- function(n_ep_pan,
                      dsl,
                      clf_model,
                      adv_model,
                      clf_optimizer,
                      adv_optimizer,
                      dev,
                      sensitive_train,
                      sensitive_test,
                      batch_size,
                      learning_rate_adv,
                      learning_rate_clf,
                      lambda,
                      verbose = TRUE,
                      monitor = TRUE) {


  if (n_ep_pan != n_ep_pan / 1 ||
      n_ep_pan < 0)
    stop("n_ep_pan must be a positive integer")
  if (typeof(clf_model) != 'closure')
    stop("provide a neural network as a clf_model")
  if (typeof(adv_model) != 'closure')
    stop("provide a neural network as a adv_model")
  if (typeof(dsl) != "list")
    stop("dsl must be list of 2 data sets and 2 data loaders from
         dataset_loader function")
  if (typeof(dsl$test_ds) != "environment")
    stop("dsl must be list of 2 data sets and 2 data loaders from
         dataset_loader function")
  if (typeof(dsl$test_ds$y) != "externalptr")
    stop("dsl must be list of 2 data sets and 2 data loaders from
         dataset_loader function")
  if (learning_rate_clf > 1 ||
      learning_rate_clf < 0)
    stop("learning_rate_clf must be between 0 and 1")
  if (learning_rate_adv > 1 ||
      learning_rate_adv < 0)
    stop("learning_rate_adv must be between 0 and 1")

  if (!typeof(clf_optimizer) == "environment")
    stop("clf_optimizer must be an optimizer used for classificator pretrain")
  if (!typeof(adv_optimizer) == "environment")
    stop("adv_optimizer must be an optimizer used for adversarials pretrain")

  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")
  if (!is.vector(sensitive_test))
    stop("sensitive_test must be a vector")
  if (!is.vector(sensitive_train))
    stop("sensitive_train must be a vector")

  if (!is.logical(verbose) ||
      !is.logical(monitor))
    stop("verbose and monitor must be logical")

  if (!is.numeric(lambda))
    stop("lambda must be numeric")

  batch_iter <- 2
  adversary_losses <- c()
  classifier_losses <- c()

  if (monitor) {
    STP <- c()
    adversary_acc <- c()
    classifier_acc <- c()
  }

  adv_model$train()
  clf_model$train()

  for (epoch in 1:n_ep_pan) {

    verbose_cat(sprintf("PAN epoch %d \n", epoch), verbose)
    train_dl <- dataloader(dsl$train_ds, batch_size = dsl$train_ds$.length(),
                                  shuffle = FALSE)
    iter <- train_dl$.iter()
    b <- iter$.next()
    output <- clf_model(b$x_cont$to(device = dev))
    preds <- output[, 2]$to(device = "cpu")

    train_x <- as.numeric(preds)
    train_x <- matrix(train_x, ncol = 1)
    train_y <- sensitive_train

    adv_dsl <- dataset_loader(train_x, train_y, train_x, train_y, batch_size,
                              dev)

    train_losses <- c()
    coro::loop(for (b in adv_dsl$train_dl) {
      output <- adv_model(b$x_cont$to(device = dev))
      loss <- torch_mul(nnf_cross_entropy(output, b$y$to(device = dev)), lambda)
      adv_optimizer$zero_grad()
      loss$backward()
      adv_optimizer$step()
      train_losses <- c(train_losses, loss$item())

    })
    #we're training the classifier on a single minibatch to cheat an adversarial
    iter <- dsl$train_dl$.iter()
    iterator <- adv_dsl$train_dl$.iter()

    for (i in 1:batch_iter) {
      iter$.next()
      iterator$.next()
    }

    batch_iter = ((batch_iter + 1) %% dsl$train_dl$.length()) + 1

    b          <- iter$.next()
    clf_output <- clf_model(b$x_cont$to(device = dev))
    preds      <- clf_output[, 2]$to(device = "cpu")
    preds      <- torch_reshape(preds, list(batch_size, 1))
    batch      <- iterator$.next()
    output     <- adv_model(preds$to(device = dev))
    loss       <- torch_mul(nnf_cross_entropy(output, batch$y$to(device = dev)),
                            lambda)
    clf_loss   <- torch_sub(nnf_cross_entropy(clf_output, b$y$to(device = dev)),
                            loss)
    clf_optimizer$zero_grad()
    clf_loss$backward()
    clf_optimizer$step()

    adversary_losses <- c(adversary_losses, mean(train_losses))
    # user wants to calculate and plot monitor values
    if(monitor) {
      acc <- eval_accuracy(adv_model, adv_dsl$test_ds, dev)
      adversary_acc <- c(adversary_acc, acc)

      cacc <- eval_accuracy(clf_model, dsl$test_ds, dev)
      classifier_acc <- c(classifier_acc, cacc)

      stp <- calc_STP(clf_model, dsl$test_ds, sensitive_test, dev)
      STP <- c(STP, stp)

      verbose_cat(sprintf("Classifier at epoch %d:training loss: %3.3f, accuracy: %3.3f\n",
                          epoch,clf_loss$item(),cacc),
                  verbose)
      verbose_cat(sprintf("Adversary at epoch %d: training loss: %3.3f,accuracy: %3.3f, STPR: %3.3f\n",
                          epoch, mean(train_losses),acc,stp),
                  verbose)
    } else{

      verbose_cat(sprintf("Adversary at epoch %d: training loss: %3.3f",
                          epoch, mean(train_losses)),
                  verbose)
    }

  }

  if(monitor) {

    monitoring <- list(
      "STP" = STP,
      "adversary_acc" = adversary_acc,
      "classifier_acc" = classifier_acc,
      "adversary_losses" = adversary_losses
    )
    return(monitoring)
  } else{
    return(NULL)
  }
}
