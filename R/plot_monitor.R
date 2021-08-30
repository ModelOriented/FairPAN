#' Plots monitored data
#'
#' Plots visualizations of monitor metrics (STP ratio, adv accuracy, adv losses
#' and classifier losses) epoch by epoch. It is useful for the monitoring of
#' the learning process, thus the user can see if everything works properly.
#' To use this, the user has to set monitor on during the fair_train process.
#'
#' @param STP double, vector of Statistical Parity ratio value
#' @param adversary_acc double, vector containing adversarial models accuracy
#' for each epoch
#' @param adversary_losses double, vector of adversaries losses
#' @param classifier_acc double, vector of adversaries accuracy
#' @param patchwork logical, if TRUE it plots all 4 plots into 2x2 matrix, with FALSE
#' plots every plot singularly
#'
#' @return NULL - plots the visualizations
#' @export
#'
#' @examples
#'
#' # presaved monitoring data
#' monitor2 <- torch_load(system.file("extdata","monitoring2",package="fairpan"))
#'
#' plot_monitor(monitor2$STP ,monitor2$adversary_acc, monitor2$classifier_acc,
#'             monitor2$adversary_losses)
#'
#' @import ggplot2
#'
plot_monitor <- function(STP = NULL,
                         adversary_acc = NULL,
                         adversary_losses = NULL,
                         classifier_acc = NULL,
                         patchwork = TRUE) {

  if (!is.null(STP) && !is.vector(STP))
    stop("STP must be a double vector")
  if (!is.null(adversary_acc) && !is.vector(adversary_acc))
    stop("adversary_acc must be a double vector")
  if (!is.null(adversary_losses) && !is.vector(adversary_losses))
    stop("adversary_losses must be a double vector")
  if (!is.null(classifier_acc) && !is.vector(classifier_acc))
    stop("classifier_acc must be a double vector")
  if (!is.logical(patchwork))
    stop("patchwork must be logical")
  # Creation of dataset for plots
  stats <- data.frame(STP, adversary_acc, adversary_losses, classifier_acc)

  STP_plot      <- NULL
  adv_acc_plot  <- NULL
  adv_loss_plot <- NULL
  clf_acc_plot  <- NULL

  if (patchwork){

    STP_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = STP), color = "#4378bf") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "STP ratio",
           title = "STP ratio") +
      theme(
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 20)
      )

    adv_acc_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = adversary_acc), color = "#4378bf") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "Accuracy", title =
             "Adversarial accuracy ") +
      theme(
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 20)
      )

    adv_loss_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = adversary_losses), color = "#4378bf") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "Loss", title = "Adversarial loss ") +
      theme(
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 20)
      )

    clf_acc_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
      geom_line(aes(y = classifier_acc), color = "#4378bf") +
      DALEX::theme_drwhy() +
      labs(x = 'Number of epochs', y = "Accuracy", title =
             "Classifier accuracy ") +
      theme(
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 20)
      )

    gridExtra::grid.arrange(STP_plot, adv_acc_plot, adv_loss_plot, clf_acc_plot, ncol=2)

  }else{
    if (!is.null(STP)) {
      STP_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
        geom_line(aes(y = STP), color = "#4378bf") +
        DALEX::theme_drwhy() +
        labs(x = 'Number of epochs', y = "STP ratio",
             title = "STP ratio") +
        theme(
          axis.text = element_text(size = 15),
          axis.title = element_text(size = 15),
          plot.title = element_text(size = 20)
        )
      plot(STP_plot)
    }

    if (!is.null(adversary_acc)) {
      adv_acc_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
        geom_line(aes(y = adversary_acc), color = "#4378bf") +
        DALEX::theme_drwhy() +
        labs(x = 'Number of epochs', y = "Accuracy", title =
               "Adversarial accuracy ") +
        theme(
          axis.text = element_text(size = 15),
          axis.title = element_text(size = 15),
          plot.title = element_text(size = 20)
        )
      plot(adv_acc_plot)
    }

    if (!is.null(adversary_losses)) {
      adv_loss_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
        geom_line(aes(y = adversary_losses), color = "#4378bf") +
        DALEX::theme_drwhy() +
        labs(x = 'Number of epochs', y = "Loss", title = "Adversarial loss ") +
        theme(
          axis.text = element_text(size = 15),
          axis.title = element_text(size = 15),
          plot.title = element_text(size = 20)
        )
      plot(adv_loss_plot)
    }

    if (!is.null(classifier_acc)) {
      clf_acc_plot <- ggplot(stats, aes(x = 1:nrow(stats))) +
        geom_line(aes(y = classifier_acc), color = "#4378bf") +
        DALEX::theme_drwhy() +
        labs(x = 'Number of epochs', y = "Accuracy", title =
               "Classifier accuracy ") +
        theme(
          axis.text = element_text(size = 15),
          axis.title = element_text(size = 15),
          plot.title = element_text(size = 20)
        )
      plot(clf_acc_plot)
    }
  }

}
