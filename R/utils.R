#' Custom print function
#'
#' Prints the provided string if verbose is TRUE.
#'
#' @param string character (string) to print
#' @param verbose logical value indicating whether we want to print the string
#' or not
#'
#' @return NULL
#'
verbose_cat <- function(string, verbose = TRUE) {
  if (verbose) {
    cat(string)
  }
}
