test_that("test-utils", {
  first <- NULL
  second <-NULL
  first <- verbose_cat("Hello","verbose"=TRUE)
  second <- verbose_cat("Hello","verbose"=FALSE)
  expect_true(is.null(verbose_cat("Hello","verbose"=FALSE)))
  expect_output(verbose_cat("Hello","verbose"=TRUE)=="Hello")
})
