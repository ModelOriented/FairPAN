#' Provides fairmodels object for selected model/s
#'
#' The functions below provide fairmodels object for one, two or three neural network
#' models with the usage of DALEX explainer
#'
#' @param target numerical target of classification task
#' @param model first model we want to explain
#' @param model2 second model we want to explain
#' @param model3 third model we want to explain
#' @param data_set numerical table of predictors
#' @param protected numerical vector of sensitive variables
#' @param privileged string label of privileged class in protected
#'
#' @return fobject
#' @export
#'
#' @examples
Single_explainer <- function(target,model,data_set,protected,privileged){

  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,test_y,data_scaled_test,test_y)$test_ds)
    pp[,2]
  }
  aps_model_exp <- DALEX::explain(model, data = data_set, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification')
  fobject <- fairness_check(aps_model_exp,
                            protected = protected,
                            privileged = privileged)
  return(fobject)
}

Triple_explainer <- function(target,model,model2,model3,data_set,protected,privileged){

  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,test_y,data_scaled_test,test_y)$test_ds)
    pp[,2]
  }
  aps_model_exp <- DALEX::explain(label ='PAN',model, data = data_set, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification')
  aps_model_exp2 <- DALEX::explain(label ='pretrain',model2, data = data_set, y = y_numeric,
                                   predict_function = custom_predict,
                                   type = 'classification')
  aps_model_exp3 <- DALEX::explain(label ='classifier_only',model3, data = data_set, y = y_numeric,
                                   predict_function = custom_predict,
                                   type = 'classification')
  fobject <- fairness_check(aps_model_exp,aps_model_exp2,aps_model_exp3,
                            protected = protected,
                            privileged = privileged)
  return(fobject)
}

Dual_explainer <- function(target,model,model2,data_set,protected,privileged){

  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,test_y,data_scaled_test,test_y)$test_ds)
    pp[,2]
  }
  aps_model_exp <- DALEX::explain(label ='classfier_only',model, data = data_set, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification')
  aps_model_exp2 <- DALEX::explain(label ='pretrain',model2, data = data_set, y = y_numeric,
                                   predict_function = custom_predict,
                                   type = 'classification')
  fobject <- fairness_check(aps_model_exp,aps_model_exp2,
                            protected = protected,
                            privileged = privileged)
  return(fobject)
}
