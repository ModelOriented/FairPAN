# FairPAN
<!-- badges: start -->
[![R-CMD-check](https://github.com/ModelOriented/FairPAN/workflows/R-CMD-check/badge.svg)](https://github.com/ModelOriented/FairPAN/actions)
<!-- badges: end -->
<!-- badges: start -->
[![Codecov test coverage](https://codecov.io/gh/ModelOriented/FairPAN/branch/master/graph/badge.svg)](https://codecov.io/gh/ModelOriented/FairPAN?branch=master)
<!-- badges: end -->

## Overview

Have you just created a model which is biased against some subgroup? Or have
you just tried to fight the bias, but models performance dropped significantly?
Use FairPAN to create neural network model that provides fair predictions and
achieves outstanding performance! With `pretrain()` you can create or provide
your own neural networks and then use them in `fair_train()` to achieve fair
outcomes. R package FairPAN additionally allows you to use lots of 
[DALEX](https://github.com/ModelOriented/DALEX) 
and [fairmodels](https://github.com/ModelOriented/fairmodels)
functions such as `DALEX::model_performance()` or `fairmodels::fairness_check()`.
*If you have problems with the training process remember to use monitor parameter and plot_monitor function for parameter adjustments*

## Installation

Install the developer version from GitHub:

```
devtools::install_github("ModelOriented/FairPAN")
```
## Example

Achieve fairness and save performance!

```

library(FairPAN)

adult <- fairmodels::adult

# ------------ step 1 - prepare data  -----------------

data <- preprocess( data = adult,
                    target_name = "salary",
                    sensitive_name = "sex",
                    privileged = "Male",
                    discriminated = "Female",
                    drop_also = c("race"),
                    sample = 0.02,
                    train_size = 0.6,
                    test_size = 0.4,
                    validation_size = 0,
                    seed = 7
)

dev <- "cpu"

dsl <- dataset_loader(train_x = data$train_x,
                      train_y = data$train_y,
                      test_x = data$test_x,
                      test_y = data$test_y,
                      batch_size = 5,
                      dev = dev
)

# ------------ step 2 - create and pretrain models  -----------------

models <- pretrain(clf_model = NULL,
                   adv_model = NULL,
                   clf_optimizer = NULL,
                   trained = FALSE,
                   train_x = data$train_x,
                   train_y = data$train_y,
                   sensitive_train = data$sensitive_train,
                   sensitive_test = data$sensitive_test,
                   batch_size = 5,
                   partition = 0.6,
                   neurons_clf = c(32, 32, 32),
                   neurons_adv = c(32, 32, 32),
                   dimension_clf = 2,
                   dimension_adv = 1,
                   learning_rate_clf = 0.001,
                   learning_rate_adv = 0.001,
                   n_ep_preclf = 10,
                   n_ep_preadv = 10,
                   dsl = dsl,
                   dev = dev,
                   verbose = TRUE,
                   monitor = TRUE
)

# ------------ step 3 - train for fairness  -----------------

monitor <- fair_train( n_ep_pan = 17,
                       dsl = dsl,
                       clf_model = models$clf_model,
                       adv_model = models$adv_model, 
                       clf_optimizer = models$clf_optimizer,
                       adv_optimizer = models$adv_optimizer,
                       dev = dev,
                       sensitive_train = data$sensitive_train,
                       sensitive_test = data$sensitive_test,  
                       batch_size = 5,   
                       learning_rate_adv = 0.001,  
                       learning_rate_clf = 0.001, 
                       lambda = 130,
                       verbose = TRUE,
                       monitor = TRUE
)

# ------------ step 4 - prepare outcomes and plot them  -----------------

plot_monitor(STP = monitor$STP,
             adversary_acc = monitor$adversary_acc,
             adversary_losses = monitor$adversary_losses,
             classifier_acc = monitor$classifier_acc)

exp_clf <- explain_PAN(y = data$test_y,
                       model = models$clf_model,
                       label = "PAN",
                       data = data$data_test,
                       data_scaled = data$data_scaled_test,
                       batch_size = 5,
                       dev = dev,
                       verbose = TRUE
)

fobject <- fairmodels::fairness_check(exp_PAN,
                            protected = data$protected_test,
                            privileged = "Male",
                            verbose = TRUE)
plot(fobject)

```

## Fair training is flexible

`pretrain` function has optional parameters:

* clf_model      nn_module describing classifiers neural network architecture

* adv_model      nn_module describing adversaries neural network architecture

* clf_optimizer  torch object providing classifier optimizer from pretrain

* trained        settles whether clf_model is trained or not

which enables users to provide their own and even pretrained neural network
models

On the other hand, you can use FairPAN package from the very beginning starting
from data preprocessing with `preprocess()` function which provides every
dataset that you will need for provided features.

## Proper evaluation

Although there are many metrics that measure fairness, our method focuses
on optimizing *Statistical Parity ratio* ( (TP+FP)/(TP+FP+TN+FN) ) which 
describes the similarity between distributions of privileged and discriminated 
variables.
