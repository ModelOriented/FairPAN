

plot_monitor <- function(STP = NULL,adversary_acc= NULL,adversary_losses= NULL,classifier_acc= NULL){

  stats<-data.table(STP,adversary_acc,adversary_losses,classifier_acc)
  if(!is.null(STP)){
    ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = STP), color = "darkred") +
      theme_drwhy() +
      labs(x='Number of epochs', y="STPR", title = "Value of STPR by the number of epochs ") +
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
  }

  if(!is.null(adversary_acc)){
    ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = adversary_acc), color="darkblue") +
      theme_drwhy() +
      labs(x='Number of epochs', y="Accuracy", title = "Adversarial accuracy ")+
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
  }

  if(!is.null(adversary_losses)){
    ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = adversary_losses), color="darkgreen") +
      theme_drwhy() +
      labs(x='Number of epochs', y="Loss", title = "Adversarial loss ")+
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
  }

  if(!is.null(classifier_acc)){
    ggplot(stats, aes(x=1:nrow(stats))) +
      geom_line(aes(y = classifier_acc), color="purple") +
      theme_drwhy() +
      labs(x='Number of epochs', y="Accuracy", title = "Classifier accuracy ")+
      theme(axis.text=element_text(size=20), axis.title=element_text(size=22), plot.title = element_text(size=26))
  }
}

plot_fairness <- function(fobject){
  plot(fobject)
}
