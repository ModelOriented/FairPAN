test_that("test-utils", {

  expect_true(is.null(verbose_cat("Hello", "verbose" = FALSE)))
  expect_output(verbose_cat("Hello", "verbose" = TRUE) == "Hello")

})
