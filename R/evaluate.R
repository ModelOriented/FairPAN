eval_accuracy <- function(model,test_ds){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont$to(device = dev))
  preds <- output$to(device = "cpu") %>% as.array()
  preds <- ifelse(preds[,1] < preds[,2], 2, 1)
  comp_df <- data.frame(preds = preds, y = b$y$to(device = "cpu") %>% as_array())
  num_correct <- sum(comp_df$preds == comp_df$y)
  num_total <- nrow(comp_df)
  accuracy <- num_correct/num_total
  return(accuracy)
}

calc_STP<-function(model,test_ds,sensitive){
  preds<-make_preds(model,test_ds)-1
  real<-(test_ds$y$to(device = "cpu") %>% as.array())-1
  sensitive<-sensitive-1
  # print(sum(real))
  TP0<-0;FP0<-0;TN0<-0;FN0<-0;Tr0<-0;Fa0<-0
  TP1<-0;FP1<-0;TN1<-0;FN1<-0;Tr1<-0;Fa1<-0

  for(i in 1:length(preds)){
    if(sensitive[i]==0){
      if(preds[i]==1 & real[i]==1){
        TP0=TP0+1
      }else if(preds[i]==1 & real[i]==0){
        FP0=FP0+1
      }else if(preds[i]==0 & real[i]==0){
        TN0=TN0+1
      }else if(preds[i]==0 & real[i]==1){
        FN0=FN0+1
      }
      if(real[i]==1){
        Tr0=Tr0+1
      }else{
        Fa0=Fa0+1
      }
    }else{
      if(preds[i]==1 & real[i]==1){
        TP1=TP1+1
      }else if(preds[i]==1 & real[i]==0){
        FP1=FP1+1
      }else if(preds[i]==0 & real[i]==0){
        TN1=TN1+1
      }else if(preds[i]==0 & real[i]==1){
        FN1=FN1+1
      }
      if(real[i]==1){
        Tr1=Tr1+1
      }else{
        Fa1=Fa1+1
      }
    }
  }
  STP0<-(TP0+FP0)/(TP0+FP0+TN0+FN0)
  STP1<-(TP1+FP1)/(TP1+FP1+TN1+FN1)
  STPR<-min(c(STP0/STP1,STP1/STP0))
  return(STPR)
}
