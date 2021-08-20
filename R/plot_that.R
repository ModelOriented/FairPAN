#' Plots monitor data
#'
#' Plots visualizations of monitor metrics (STP ratio, adv accuracy, adv losses
#' and classifier losses) epoch by epoch. It is useful for the monitoring of
#' the learning process, thus the user can see if everything works properly.
#'
#' @param STP double vector of Statistical Parity ratio value
#' @param adversary_acc double, vector containing adversarial models accuracy
#' for each epoch
#' @param adversary_losses double vector of adversaries losses
#' @param classifier_acc double vector of adversarials accuracy
#'
#' @return NULL - plots the visualizations
#' @export
#'
#' @examples
#' \dontrun{
#' monitor2 <- torch_load("./tests/zzz/monitoring2")
#' plot_monitor(monitor2$STP ,monitor2$adversary_acc, monitor2$classifier_acc,
#'              monitor2$adversary_losses)
#' }
#' @import ggplot2
#'
plot_monitor <- function(STP = NULL,
                         adversary_acc = NULL,
                         adversary_losses = NULL,
                         classifier_acc = NULL) {

  if (!is.null(STP) && !is.vector(STP))
    stop("STP must be a double vector")
  if (!is.null(adversary_acc) &&
      !is.vector(adversary_acc))
    stop("adversary_acc must be a double vector")
  if (!is.null(adversary_losses) &&
      !is.vector(adversary_losses))
    stop("adversary_losses must be a double vector")
  if (!is.null(classifier_acc) &&
      !is.vector(classifier_acc))
    stop("classifier_acc must be a double vector")

  stats <- data.frame(STP, adversary_acc, adversary_losses, classifier_acc)

  if (!is.null(STP)) {
    STP_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = STP), color = "darkred") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "STP ratio",
           title = "STP ratio") +
      theme(
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 22),
        plot.title = element_text(size = 26)
      )
    plot(STP_plot)
  }

  if (!is.null(adversary_acc)) {
    adv_acc_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = adversary_acc), color = "darkblue") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "Accuracy", title =
             "Adversarial accuracy ") +
      theme(
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 22),
        plot.title = element_text(size = 26)
      )
    plot(adv_acc_plot)
  }

  if (!is.null(adversary_losses)) {
    adv_loss_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = adversary_losses), color = "darkgreen") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "Loss", title = "Adversarial loss ") +
      theme(
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 22),
        plot.title = element_text(size = 26)
      )
    plot(adv_loss_plot)
  }

  if (!is.null(classifier_acc)) {
    clf_acc_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = classifier_acc), color = "purple") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "Accuracy", title =
             "Classifier accuracy ") +
      theme(
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 22),
        plot.title = element_text(size = 26)
      )
    plot(clf_acc_plot)
  }
}

# Plots fobject metrics
#
# This function plots fairness metrics like Equal parity ratio and statistical
# parity ratio. The plot is provided by fairmodels.
#
# @param fobject fairness object from fairmodels
#
# @return NULL - plots the visualization
# @export
#
# @examples
# dev   <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"
#
# data("adult")
# processed <- preprocess( adult, "salary", "sex", "Male", "Female", c("race"),
#                          sample = 0.8, train_size = 0.65, test_size = 0.35,
#                          validation_size = 0, seed = 7)
#
# model1    <- torch_load("./zzz/clf1")
#
# dsl       <- dataset_loader(processed$train_x, processed$train_y,
#                             processed$test_x, processed$test_y,
#                             batch_size=5, dev=dev)
#
# exp <- Single_explainer(
#   processed$test_y,
#   model1,
#   "classifier",
#   processed$data_test,
#   processed$data_scaled_test,
#   processed$protected_test,
#   privileged = "Male",
#   batch_size = 5,
#   dev = dev,
#   verbose = FALSE
# )
#
# plot_fairness(exp)

# plot_fairness <- function(fobject) {
#
#   plot(fobject)
#
# }
