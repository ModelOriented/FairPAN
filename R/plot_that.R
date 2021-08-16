

#' Plots visualizations that show the learning process
#'
#' @param STP double vector of Statistical Parity ratio value
#' @param adversary_acc double, vector containing adversarial models accuracy for each epoch
#' @param adversary_losses double vector of adversaries losses
#' @param classifier_acc double vector of adversarials accuracy
#'
#' @return NULL - plots the visualizations
#' @export
#'
#' @examples
plot_monitor <- function(STP = NULL,adversary_acc= NULL,adversary_losses= NULL,classifier_acc= NULL){
  if(!is.null(STP) && !is.vector(STP)) stop("STP must be a double vector")
  if(!is.null(adversary_acc) && !is.vector(adversary_acc)) stop("adversary_acc must be a double vector")
  if(!is.null(adversary_losses) && !is.vector(adversary_losses)) stop("adversary_losses must be a double vector")
  if(!is.null(classifier_acc) && !is.vector(classifier_acc)) stop("classifier_acc must be a double vector")

  stats<-data.table(STP,adversary_acc,adversary_losses,classifier_acc)

  if(!is.null(STP)){
    STP_plot<-ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = STP), color = "darkred") +
      theme_drwhy() +
      labs(x='Number of epochs', y="STPR", title = "Value of STPR by the number of epochs ") +
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
    plot(STP_plot)
  }

  if(!is.null(adversary_acc)){
    adv_acc_plot <-ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = adversary_acc), color="darkblue") +
      theme_drwhy() +
      labs(x='Number of epochs', y="Accuracy", title = "Adversarial accuracy ")+
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
    plot(adv_acc_plot)
  }

  if(!is.null(adversary_losses)){
    adv_loss_plot <-ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = adversary_losses), color="darkgreen") +
      theme_drwhy() +
      labs(x='Number of epochs', y="Loss", title = "Adversarial loss ")+
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
    plot(adv_loss_plot)
  }

  if(!is.null(classifier_acc)){
    clf_acc_plot <-ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = classifier_acc), color="purple") +
      theme_drwhy() +
      labs(x='Number of epochs', y="Accuracy", title = "Classifier accuracy ")+
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
    plot(clf_acc_plot)
  }
}

#' Plots fobject metrics
#'
#' This function plots fairness metrics like Equal parity ratio and statistical parity ratio.
#' The plot is provided by fairmodels
#'
#' @param fobject fairness object from fairmodels
#'
#' @return NULL - plots the visualization
#' @export
#'
#' @examples
plot_fairness <- function(fobject){

  plot(fobject)
}
