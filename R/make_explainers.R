#' Provides fairmodels object for selected model
#'
#' The functions below provide fairmodels object for one neural network
#' model with the usage of DALEX explainer and fairmodels fairness_check.
#'
#' @param target numerical target of classification task
#' @param model the model we want to explain
#' @param model_name character providing the label (name) to first model
#' @param data_test numerical list (table) of predictors
#' @param data_scaled_test scaled matrix of numerical values representing
#' predictors
#' @param batch_size integer indicating a batch size used in dataloader.
#' @param dev device used to calculations (cpu or gpu)
#' @param verbose logical indicating if we want to print monitored outputs or
#' not. Default: TRUE.
#'
#' @return fobject - fairness object
#' @export
#'
#' @examples
#'
#'


explain_PAN <- function(target,
                        model,
                        model_name,
                        data_test,
                        data_scaled_test,
                        batch_size,
                        dev,
                        verbose = TRUE) {

  if (!is.vector(target))
    stop("target must be a vector")
  if (typeof(model) != 'closure')
    stop("models must be neural networks models")
  if (typeof(model_name) != 'character')
    stop("models names must be characters")
  if (!is.list(data_test))
    stop("data_test must be a list")
  if (!is.matrix(data_scaled_test))
    stop("data_scaled_test must be a matrix")
  if (nrow(data_scaled_test) != nrow(data_test))
    stop("data_scaled_test and data_test must have equal number of rows")
  if (batch_size != batch_size / 1)
    stop("batch size must be an integer")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")
  if (!is.logical(verbose))
    stop("verbose must be logical")

  y_numeric <- as.numeric(target) - 1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel,
                        test_ds = dataset_loader(data_scaled_test,
                                                 target,
                                                 data_scaled_test,
                                                 target,
                                                 batch_size,
                                                 dev)$test_ds, dev)
    pp[,2]
  }


  aps_model_exp <- DALEX::explain(label = model_name, model,
                                  data = data_test, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification', verbose = verbose)

  return(aps_model_exp)
}
