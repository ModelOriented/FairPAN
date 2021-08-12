#' Provides fairmodels object for selected model
#'
#' The functions below provide fairmodels object for one neural network
#' model with the usage of DALEX explainer and fairmodels fairness_check.
#'
#' @param target numerical target of classification task
#' @param model the model we want to explain
#' @param model_name character providing the label (name) to first model
#' @param data_test numerical table of predictors
#' @param protected numerical vector of sensitive variables
#' @param privileged string label of privileged class in protected
#' @param data_scaled_test scaled matrix of numerical values representing predictors
#' @param batch_size integer indicating a batch size used in dataloader.
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return fobject - fairness object
#' @export
#'
#' @examples
Single_explainer <- function(target,model,model_name,data_test,data_scaled_test,protected,privileged,batch_size,dev){
  if(!is.vector(target)) stop("target must be a vector")
  if(typeof(model)!='closure') stop("models must be neural networks models")
  if(typeof(model_name)!='character') stop("model names must be characters")
  if(nrow(data_test)!=nrow(data_scaled_test)) stop("number of rows of data_test and data_scaled_test cannot differ")
  if(nrow(data_test)!=length(protected)) stop("number of rows of data_test and length of protected must be the same")
  if(typeof(privileged)!='character') stop("privileged must be character name of label from protected")
  if(batch_size!=batch_size/1) stop("batch size must be an integer")
  if(!dev %in% c("gpu","cpu"))stop("dev must be gpu or cpu")

  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,target,data_scaled_test,target,batch_size,dev)$test_ds,dev)
    pp[,2]
  }
  #print(custom_predict(model,c(1,2,3)))

  aps_model_exp <- DALEX::explain(label=model_name,model, data = data_test, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification')

  fobject <- fairness_check(aps_model_exp,
                            protected = protected,
                            privileged = privileged)
  return(fobject)
}

#' Provides fairmodels object for selected two models
#'
#' The functions below provide fairmodels object for two neural network
#' models with the usage of DALEX explainer and fairmodels fairness_check.
#'
#' @param target numerical target of classification task
#' @param model first model we want to explain
#' @param model2 second model we want to explain
#' @param model_name character providing the label (name) to first model
#' @param model_name2 character providing the label (name) to second model
#' @param data_test numerical table of predictors
#' @param protected numerical vector of sensitive variables
#' @param privileged string label of privileged class in protected
#' @param data_scaled_test scaled matrix of numerical values representing predictors
#' @param batch_size integer indicating a batch size used in dataloader.
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return fobject - fairness object
#' @export
#'
#' @examples
Dual_explainer <- function(target,model,model2,model_name,model_name2,data_test,data_scaled_test,protected,privileged,batch_size,dev){
  if(!is.vector(target)) stop("target must be a vector")
  if(!is.matrix(train_x)) stop("train_x must be a matrix")
  if(typeof(model)!='closure' || typeof(model2)!='closure') stop("models must be neural networks models")
  if(typeof(model_name)!='character' || typeof(model_name2)!='character') stop("model names must be characters")
  if(nrow(data_test)!=nrow(data_scaled_test)) stop("number of rows of data_test and data_scaled_test cannot differ")
  if(nrow(data_test)!=length(protected)) stop("number of rows of data_test and length of protected must be the same")
  if(typeof(privileged)!='character') stop("privileged must be character name of label from protected")
  if(batch_size!=batch_size/1) stop("batch size must be an integer")
  if(!dev %in% c("gpu","cpu"))stop("dev must be gpu or cpu")
  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,target,data_scaled_test,target,batch_size,dev)$test_ds,dev)
    pp[,2]
  }
  aps_model_exp <- DALEX::explain(label =model_name,model, data = data_test, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification')
  aps_model_exp2 <- DALEX::explain(label =model_name2,model2, data = data_test, y = y_numeric,
                                   predict_function = custom_predict,
                                   type = 'classification')
  fobject <- fairness_check(aps_model_exp,aps_model_exp2,
                            protected = protected,
                            privileged = privileged)
  return(fobject)
}


#' Provides fairmodels object for selected three model
#'
#' The functions below provide fairmodels object for three neural network
#' models with the usage of DALEX explainer and fairmodels fairness_check.
#'
#' @param target numerical target of classification task
#' @param model first model we want to explain
#' @param model2 second model we want to explain
#' @param model3 third model we want to explain
#' @param model_name character providing the label (name) to first model
#' @param model_name2 character providing the label (name) to second model
#' @param model_name3 character providing the label (name) to third model
#' @param data_test numerical table of predictors
#' @param protected numerical vector of sensitive variables
#' @param privileged string label of privileged class in protected
#' @param data_scaled_test scaled matrix of numerical values representing predictors
#' @param batch_size integer indicating a batch size used in dataloader.
#' @param dev device used to calculations (cpu or gpu)
#'
#' @return fobject - fairness object
#' @export
#'
#' @examples
Triple_explainer <- function(target,model,model2,model3,model_name,model_name2,model_name3,data_test,data_scaled_test,protected,privileged,batch_size,dev){
  if(!is.vector(target)) stop("target must be a vector")
  if(!is.matrix(train_x)) stop("train_x must be a matrix")
  if(typeof(model)!='closure' || typeof(model2)!='closure' || typeof(model3)!='closure') stop("models must be neural networks models")
  if(typeof(model_name)!='character' || typeof(model_name2)!='character' || typeof(model_name3)!='character') stop("model names must be characters")
  if(nrow(data_test)!=nrow(data_scaled_test)) stop("number of rows of data_test and data_scaled_test cannot differ")
  if(nrow(data_test)!=length(protected)) stop("number of rows of data_test and length of protected must be the same")
  if(typeof(privileged)!='character') stop("privileged must be character name of label from protected")
  if(batch_size!=batch_size/1) stop("batch size must be an integer")
  if(!dev %in% c("gpu","cpu"))stop("dev must be gpu or cpu")



  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,target,data_scaled_test,target,batch_size,dev)$test_ds,dev)
    pp[,2]
  }
  aps_model_exp <- DALEX::explain(label =model_name,model, data = data_test, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification')
  aps_model_exp2 <- DALEX::explain(label =model_name2,model2, data = data_test, y = y_numeric,
                                   predict_function = custom_predict,
                                   type = 'classification')
  aps_model_exp3 <- DALEX::explain(label =model_name3,model3, data = data_test, y = y_numeric,
                                   predict_function = custom_predict,
                                   type = 'classification')
  fobject <- fairness_check(aps_model_exp,aps_model_exp2,aps_model_exp3,
                            protected = protected,
                            privileged = privileged)
  return(fobject)
}
