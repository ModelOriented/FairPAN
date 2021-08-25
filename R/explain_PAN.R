#' Creates explainer of PAN model
#'
#' This function creates DALEX explainer for PAN model (or other neural network)
#' All DALEX functions such as model_performance or
#'
#' @param y numerical target of classification task
#' @param model net, nn_module, the model we want to explain
#' @param label character providing the label (name) to first model
#' @param data numerical list (table) of predictors
#' @param data_scaled scaled matrix of numerical values representing
#' predictors
#' @param batch_size integer indicating a batch size used in dataloader.
#' @param dev device used to calculations (cpu or gpu)
#' @param verbose logical indicating if we want to print monitored outputs or
#' not. Default: TRUE.
#'
#' @return DALEX model explainer
#' @export
#'
#' @examples
#'
#' \dontrun{
#' dev <- "cpu"
#'
#' adult <- fairmodels::adult
#'
#' processed <- preprocess(
#'   adult,
#'   "salary",
#'   "sex",
#'   "Male",
#'   "Female",
#'   c("race"),
#'   sample = 0.8,
#'   train_size = 0.65,
#'   test_size = 0.35,
#'   validation_size = 0,
#'   seed = 7
#' )
#' # presaved torch model
#' model1 <- torch::torch_load(system.file("extdata","clf1",package="FairPAN"))
#'
#' dsl <- dataset_loader(processed$train_x, processed$train_y,
#'                       processed$test_x, processed$test_y,
#'                       batch_size=5, dev=dev)
#'
#' explainer <- explain_PAN(
#'   processed$test_y,
#'   model1,
#'   "classifier",
#'   processed$data,
#'   processed$data_scaled,
#'   batch_size = 5,
#'   dev = dev,
#'   verbose = FALSE
#' )
#' }
#'
#'


explain_PAN <- function(y,
                        model,
                        label,
                        data,
                        data_scaled,
                        batch_size,
                        dev,
                        verbose = TRUE) {

  if (!is.vector(y))
    stop("y must be a vector")
  if (typeof(model) != 'closure')
    stop("models must be neural networks models")
  if (typeof(label) != 'character')
    stop("label must be a character")
  if (!is.list(data))
    stop("data must be a list")
  if (!is.matrix(data_scaled))
    stop("data_scaled must be a matrix")
  if (nrow(data_scaled) != nrow(data))
    stop("data_scaled and data must have equal number of rows")
  if (batch_size != batch_size / 1)
    stop("batch size must be an integer")
  if (!dev %in% c("gpu", "cpu"))
    stop("dev must be gpu or cpu")
  if (!is.logical(verbose))
    stop("verbose must be logical")

  y_numeric <- as.numeric(y) - 1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel,
                        test_ds = dataset_loader(data_scaled,
                                                 y,
                                                 data_scaled,
                                                 y,
                                                 batch_size,
                                                 dev)$test_ds, dev)
    pp[,2]
  }


  aps_model_exp <- DALEX::explain(label = label, model,
                                  data = data, y = y_numeric,
                                  predict_function = custom_predict,
                                  type = 'classification', verbose = verbose)

  return(aps_model_exp)
}
