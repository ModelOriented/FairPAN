test_that("test-plot_monitor", {

  setwd("..")

  monitor2 <- torch::torch_load("./zzz/monitoring2")
  monitor3 <- torch::torch_load("./zzz/monitoring3")

  plot1 <- plot_monitor(monitor2$STP, monitor2$adversary_acc,
                        monitor2$classifier_acc, monitor2$adversary_losses)
  plot2 <- plot_monitor(monitor3$STP, monitor3$adversary_acc,
                        monitor3$classifier_acc, monitor3$adversary_losses)

  plot3 <- plot_monitor(monitor2$STP, monitor2$adversary_acc,
                        monitor2$classifier_acc, monitor2$adversary_losses,
                        patchwork = FALSE)
  plot4 <- plot_monitor(monitor3$STP, monitor3$adversary_acc,
                        monitor3$classifier_acc, monitor3$adversary_losses,
                        patchwork = FALSE)

  expect_s3_class(plot1, "gtable")
  expect_s3_class(plot2, "gtable")
  expect_s3_class(plot3, "ggplot")
  expect_s3_class(plot4, "ggplot")

  expect_error(
    plot_monitor(
      as.matrix(monitor2$STP),
      monitor2$adversary_acc,
      monitor2$classifier_acc,
      monitor2$adversary_losses
      )

  )

  expect_error(
    plot_monitor(
      monitor2$STP,
      as.matrix(monitor2$adversary_acc),
      monitor2$classifier_acc,
      monitor2$adversary_losses
      )

  )

  expect_error(
    plot_monitor(
      monitor2$STP,
      monitor2$adversary_acc,
      as.matrix(monitor2$classifier_acc),
      monitor2$adversary_losses
      )

  )

  expect_error(
    plot_monitor(
      monitor2$STP,
      monitor2$adversary_acc,
      monitor2$classifier_acc,
      as.matrix(monitor2$adversary_losses)
      )

  )

})
