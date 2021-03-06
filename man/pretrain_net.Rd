% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pretrain_net.R
\name{pretrain_net}
\alias{pretrain_net}
\title{Pretrains the neural network}
\usage{
pretrain_net(
  n_epochs = 15,
  model,
  dsl,
  model_type,
  learning_rate = 0.001,
  sensitive_test,
  dev,
  verbose = TRUE,
  monitor = TRUE
)
}
\arguments{
\item{n_epochs}{integer setting number of epochs for training. Default: 15}

\item{model}{net, nn_module, neural network model we want to train}

\item{dsl}{dataset_loader object for the training}

\item{model_type}{indicates which model we train (0 for preadversarial, 1 for
preclassifier, 2 for classifier only)}

\item{learning_rate}{float from [0,1] setting learning rate for model.
Default: 0.001}

\item{sensitive_test}{test vector for sensitive variable used to calculate
STP}

\item{dev}{device used to calculations (cpu or gpu)}

\item{verbose}{logical indicating if we want to print monitored outputs or
not}

\item{monitor}{logical indicating if we want to monitor the learning process
or not (monitoring tends to slow down the training process, but provides
some useful info to adjust parameters and training process)}
}
\value{
list(train_loss,test_loss,optimizer)
}
\description{
Pretrain for both the classifier and adversarial model. You can select which
model it is by setting model_type parameter (it customizes the prints properly).
Pretrain can also collect more data to training analysis and print it out
which is useful for monitoring the learning process and making adjustments.
}
\examples{
\dontrun{
dev <- "cpu"

# presaved output of preprocess function
processed <- torch_load(system.file("extdata","processed",package="fairpan"))
dsl <- dataset_loader(processed$train_x, processed$train_y, processed$test_x,
                      processed$test_y, batch_size=5, dev=dev)

# presaved torch model
model <- torch_load(system.file("extdata","clf1",package="fairpan"))
pretrain_net(
  n_epochs = 1,
  model = model,
  dsl = dsl,
  model_type = 1,
  learning_rate = 0.001,
  sensitive_test = processed$sensitive_test,
  dev=dev,
  verbose = TRUE,
  monitor = TRUE
)
}

}
