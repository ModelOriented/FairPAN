#' Title
#'
#' @param n_ep_pan
#' @param dsl
#' @param clf_model
#' @param adv_model
#' @param dev
#' @param sensitive_train
#' @param sensitive_test
#' @param batch_size
#' @param learning_rate_adv
#' @param learning_rate_clf
#' @param lambda
#'
#' @return
#' @export
#'
#' @examples
fair_train <- function(n_ep_pan=50, dsl, clf_model, adv_model, dev, sensitive_train,
                       sensitive_test, batch_size=50, learning_rate_adv=0.001,
                       learning_rate_clf=0.001, lambda=130){

  train_PAN(n_ep_pan, dsl, clf_model, adv_model, dev, sensitive_train,
            sensitive_test, batch_size, learning_rate_adv,
            learning_rate_clf, lambda)

}
