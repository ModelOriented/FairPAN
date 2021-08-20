## ----import-adult-------------------------------------------------------------
library(FairPAN)
adult <- fairmodels::adult
head(adult)

## ----preprocess---------------------------------------------------------------
data <- preprocess( data = adult,               # dataset 
                    target_name = "salary",     # name of target column
                    sensitive_name = "sex",     # name of sensitive column
                    privileged = "Male",        # level of privileged class
                    discriminated = "Female",   # level of discriminated class
                    drop_also = c("race"),      # columns to drop (perhaps 
                                                # other sensitive variable)
                    sample = 0.04,              # sample size from dataset
                    train_size = 0.6,           # size of train set
                    test_size = 0.4,            # size of test set
                    validation_size = 0,        # size of validation set
                    seed = 7                    # seed for reproduction.
)

head(data$train_x,2)
head(data$train_y,2)
head(data$sensitive_train,2)
head(data$test_x,2)
head(data$test_y,2)
head(data$sensitive_test,2)
head(data$data_scaled_test,2)
head(data$data_test,2)
head(data$protected_test,2)


## ----dataset_loader-----------------------------------------------------------
dev <- if (torch::cuda_is_available()) torch_device("cuda:0") else "cpu"

dsl <- dataset_loader(train_x = data$train_x,
                      train_y = data$train_y,
                      test_x = data$test_x,
                      test_y = data$test_y,
                      batch_size = 5,
                      dev = dev
)
print(dsl$train_dl$.iter()$.next())

## ----auto-pretrain------------------------------------------------------------

models <- pretrain(clf_model = NULL,                       # classifier model
                   adv_model = NULL,                       # adversarial model
                   clf_optimizer = NULL,                   # classifiers optimizer
                   trained = FALSE,                        # indicates whether provided classifier is 
                                                           # trained
                   
                   train_x = data$train_x,                 # train predictors
                   train_y = data$train_y,                 # train target
                   sensitive_train = data$sensitive_train, # train sensitives
                   sensitive_test = data$sensitive_test,   # test sensitives
                   
                   batch_size = 5,                         # inner dataset_loader batch size
                   partition = 0.6,                        # partition for inner adversaries 
                                                           # dataset_loader preparation
                   
                   neurons_clf = c(32, 32, 32),            # classifiers neural architecture
                   neurons_adv = c(32, 32, 32),            # adversaries neural architecture
                   dimension_clf = 2,                      # dimension for classifier (always set 2)
                   dimension_adv = 1,                      # dimension for adversarial (always set 1)
                   learning_rate_clf = 0.001,              # learning rate of classifier
                   learning_rate_adv = 0.001,              # learning rate of adversarial
                   n_ep_preclf = 5,                        # number of epochs for classifier pretrain
                   n_ep_preadv = 10,                       # number of epochs for adversarial pretrain
                   
                   dsl = dsl,                              # dataset_loader
                   dev = dev,                              # computational device
                   verbose = TRUE,                         # if TRUE prints metrics
                   monitor = TRUE                          # if TRUE aquires more data ( also to print)
)


## ----advanced-pretrain--------------------------------------------------------

clf <- create_model(train_x = data$train_x,                # train predictors
                    train_y = data$train_y,                # train target 
                    neurons = c(32,32,32),                 # models neural architecture
                    dimensions = 2                         # dimension for model (always set 2 for 
                                                           # classifier 1 for adversary)
)

opt <- pretrain_net(n_epochs = 5,                          # number of epochs for model pretrain
                    model = clf,                           # neural network model
                    dsl = dsl,                             # dataset_loader
                    model_type = 1,                        # model type (1 means precalssifer)
                    learning_rate = 0.001,                 # learning rate of classifier
                    sensitive_test = data$sensitive_test,  # test sensitives
                    dev = dev,                             # computational device
                    verbose = TRUE,                        # if TRUE prints metrics
                    monitor = TRUE                         # if TRUE aquires more data ( also to print)
)

print(opt$optimizer)

clf_optimizer <- opt$optimizer
    
models <- pretrain(clf_model = clf,                        # classifier model
                   adv_model = NULL,                       # adversarial model
                   clf_optimizer = clf_optimizer,          # classifiers optimizer
                   trained = TRUE,                         # indicates whether provided classifier is 
                                                           # trained
                   train_x = data$train_x,                 # train predictors
                   train_y = data$train_y,                 # train target
                   sensitive_train = data$sensitive_train, # train sensitives
                   sensitive_test = data$sensitive_test,   # test sensitives
                   batch_size = 5,                         # inner dataset_loader batch size
                   partition = 0.6,                        # partition for inner adversaries 
                                                           # dataset_loader preparation
                   neurons_clf = c(32, 32, 32),            # classifiers neural architecture
                   neurons_adv = c(32, 32, 32),            # adversaries neural architecture
                   dimension_clf = 2,                      # dimension for classifier (always set 2)
                   dimension_adv = 1,                      # dimension for adversarial (always set 1)
                   learning_rate_clf = 0.001,              # learning rate of classifier
                   learning_rate_adv = 0.001,              # learning rate of adversarial
                   n_ep_preclf = 5,                        # number of epochs for classifier pretrain
                   n_ep_preadv = 10,                       # number of epochs for adversarial pretrain
                   dsl = dsl,                              # dataset_loader
                   dev = dev,                              # computational device
                   verbose = TRUE,                         # if TRUE prints metrics
                   monitor = TRUE                          # if TRUE aquires more data ( also to print)
)

## ----explain_clf--------------------------------------------------------------
exp_clf <- explain_PAN(target = data$test_y,                     # test target
                       model = models$clf_model,                 # classifier model
                       model_name = "Classifier",                # classifiers name
                       data_test = data$data_test,               # original data for test
                       data_scaled_test = data$data_scaled_test, # scaled numeric data for test
                       batch_size = 5,                           # batch_size used in dataset_loader
                       dev = dev,                                # computational device
                       verbose = TRUE                            # if TRUE prints monitor info
)

## ----fairtrain----------------------------------------------------------------
monitor <- fair_train( n_ep_pan = 30,                           # number of epochs for pan training
                       dsl = dsl,                               # dataset_loader
                       
                       clf_model = models$clf_model,            # classifier model
                       adv_model = models$adv_model,            # adv model
                       clf_optimizer = models$clf_optimizer,    # classifiers optimizer
                       adv_optimizer = models$adv_optimizer,    # adversaries optimizer
                       
                       dev = dev,                               # computational device
                       sensitive_train = data$sensitive_train,  # train sensitives
                       sensitive_test = data$sensitive_test,    # test sensitives
                       
                       batch_size = 5,                          # inner dataset_loader batch size
                       learning_rate_adv = 0.001,               # learning rate of adversarial
                       learning_rate_clf = 0.001,               # learning rate of classifier
                       lambda = 130,                            # train controlling parameter (the 
                                                                # bigger the better STPR results)
                       
                       verbose = FALSE,                         # if TRUE prints metrics
                       monitor = TRUE                           # if TRUE training collects 4 metrics 
                                                                # throughout the epochs
)
monitor

