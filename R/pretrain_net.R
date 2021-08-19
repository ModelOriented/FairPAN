#' Pretrains the neural network
#'
#' Pretrain for both the classifier and adversarial model. You can select which
#' model it is by setting model_type parameter
#'
#' @param n_epochs integer setting number of epochs for training. Default: 15
#' @param model neural network model we want to train
#' @param dsl dataset_loader object for the training
#' @param model_type indicates which model we train (0 for preadversarial, 1 for
#' preclassifier, 2 for classifier only)
#' @param learning_rate float from [0,1] setting learning rate for model.
#' Default: 0.001
#' @param sensitive_test test vector for sensitive variable used to calculate
#' STP
#' @param dev device used to calculations (cpu or gpu)
#' @param verbose logical indicating if we want to print monitored outputs or
#' not
#' @param monitor logical indicating if we want to monitor the learning process
#' or not (monitoring tends to slow down the training proccess, but provides
#' some useful info to adjust parameters and training process)
#'
#' @return list(train_loss,test_loss,optimizer)
#' @export
#'
#' @examples
#' \dontrun{
#' dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"
#' processed <- torch_load("./tests/zzz/processed")
#' dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
#'                       processed$test_y, batch_size=5, dev=dev)
#' model <- torch_load("./tests/zzz/clf1")
#' pretrain_net(
#'   n_epochs = 3,
#'   model = model,
#'   dsl = dsl,
#'   model_type = 2,
#'   learning_rate = 0.001,
#'   sensitive_test = processed$sensitive_test,
#'   dev=dev,
#'   verbose = TRUE,
#'   monitor = TRUE
#' )
#' }
#'
pretrain_net <- function(n_epochs = 15,
                         model,
                         dsl,
                         model_type,
                         learning_rate = 0.001,
                         sensitive_test,
                         dev,
                         verbose = TRUE,
                         monitor = TRUE) {


  if (n_epochs != n_epochs / 1 ||
      n_epochs < 0)
    stop("n_epochs must be a positive integer")
  if (typeof(model) != 'closure')
    stop("provide a neural network as a model")
  if (typeof(dsl) != "list")
    stop("dsl must be list of 2 data sets and 2 data loaders from
         dataset_loader function")
  if (typeof(dsl$test_ds) != "environment")
    stop("dsl must be list of 2 data sets and 2 data loaders from
         dataset_loader function")
  if (typeof(dsl$test_ds$y) != "externalptr")
    stop("dsl must be list of 2 data sets and 2 data loaders from
         dataset_loader function")
  if (learning_rate > 1 ||
      learning_rate < 0)
    stop("learning_rate must be between 0 and 1")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")
  if (!is.vector(sensitive_test))
    stop("sensitive_test must be a vector")
  if (!is.logical(verbose) ||
      !is.logical(monitor))
    stop("verbose and monitor must be logical")

    optimizer <- torch::optim_adam(model$parameters, lr = learning_rate)

    calc_loss <- function(output, batch) {
      loss <- torch::nnf_cross_entropy(output, batch)
      return(loss)
    }

    train_eval <- function(model, dsl, optimizer, dev) {
      model$train()
      train_losses <- c()
      coro::loop(for (b in dsl$train_dl) {
        optimizer$zero_grad()
        output <- model(b$x_cont$to(device = dev))
        loss <- calc_loss(output, b$y$to(device = dev))
        loss$backward()
        optimizer$step()
        train_losses <- c(train_losses, loss$item())
      })
      model$eval()
      valid_losses <- c()
      coro::loop(for (b in dsl$test_dl) {
        output <- model(b$x_cont$to(device = dev))
        loss <- calc_loss(output, b$y$to(device = dev))
        valid_losses <- c(valid_losses, loss$item())
      })

      return(list( "train_loss" = mean(train_losses),
                   "test_loss" = mean(valid_losses)))
    }

    if (model_type == 0) {
      for (epoch in 1:n_epochs) {
        losses <- train_eval(model, dsl, optimizer, dev)
        if (monitor) {
          acc <- eval_accuracy(model, dsl$test_ds, dev)
          verbose_cat(
            sprintf(
              "Preadversary at epoch %d: training loss: %3.3f, validation: %3.3f, accuracy: %3.3f\n",
              epoch, losses$train_loss, losses$test_loss, acc
            ),
            verbose = verbose
          )
        } else{
          verbose_cat(
            sprintf(
              "Preadversary at epoch %d: training loss: %3.3f,
              validation: %3.3f\n",
              epoch, losses$train_loss, losses$test_loss
            ),
            verbose = verbose
          )
        }
      }
    }
    if (model_type == 1) {
      for (epoch in 1:n_epochs) {
        # TODO seq_len
        losses <- train_eval(model, dsl, optimizer, dev)
        if (monitor) {
          acc <- eval_accuracy(model, dsl$test_ds, dev)
          stp <- calc_STP(model, dsl$test_ds, sensitive_test, dev)
          verbose_cat(
            sprintf(
              "Preclassifier at epoch %d: training loss: %3.3f, validation: %3.3f, accuracy: %3.3f, STPR: %3.3f\n",
              epoch, losses$train_loss, losses$test_loss, acc, stp
            ),
            verbose = verbose
          )
        } else{
          verbose_cat(
            sprintf(
              "Preclassifier at epoch %d: training loss: %3.3f,
              validation: %3.3f\n",
              epoch, losses$train_loss, losses$test_loss
            ),
            verbose = verbose
          )
        }
      }
    }
    if (model_type == 2) {
      for (epoch in 1:n_epochs) {
        losses <- train_eval(model, dsl, optimizer, dev)
        if (monitor) {
          acc <- eval_accuracy(model, dsl$test_ds, dev)
          stp <- calc_STP(model, dsl$test_ds, sensitive_test, dev)
          verbose_cat(
            sprintf(
              "Classifier only at epoch %d: training loss: %3.3f, validation: %3.3f, accuracy: %3.3f, STPR: %3.3f\n",
              epoch, losses$train_loss, losses$test_loss, acc, stp
            ),
            verbose = verbose
          )
        } else{
          verbose_cat(
            sprintf(
              "Classifier only at epoch %d: training loss: %3.3f, validation: %3.3f\n",
              epoch, losses$train_loss, losses$test_loss
            ),
            verbose = verbose
          )
        }
      }
    }
  return(list("train_loss"=losses$train_loss, "test_loss"=losses$test_loss,
              "optimizer"=optimizer))
}
