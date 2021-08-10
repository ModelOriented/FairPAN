#' Custom print function
#'
#' Prints the string if verbose is TRUE
#'
#' @param string character to print
#' @param verbose logical indicating if we want to print the string or not
#'
#' @return NULL
verbose_cat<-function(string, verbose=TRUE){
  if(verbose){
    cat(string)
  }
}
