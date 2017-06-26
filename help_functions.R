# Function that gets the data and does the factor and date transformations# 

get.data <- function(data){ 
  if(data=="known"){
    
  
  df_known<- read.csv2( "assignment_BADS_WS1617_known.csv", header = TRUE, sep = ",", dec= ":"  )
  df_known$order_date <- ymd(df_known$order_date) 
  df_known$account_creation_date<- ymd(df_known$account_creation_date)
  df_known$deliverydate_actual <- ymd(df_known$deliverydate_actual) #NA values !
  df_known$deliverydate_estimated <- ymd(df_known$deliverydate_estimated)
  df_known$title <- factor(df_known$title, labels = c("No","Yes"))
  df_known$giftwrapping   <- factor(df_known$giftwrapping,labels =c("No", "Yes"))
  df_known$coupon <- factor(df_known$coupon, labels = c("No", "Yes"))
  df_known$newsletter <- factor(df_known$newsletter, labels = c("No", "Yes"))
  df_known$delivery <- factor(df_known$delivery, labels = c("No", "Yes"))
  df_known$referrer <- factor(df_known$referrer, labels = c("No", "Yes"))
  df_known$cost_shipping <- factor(df_known$cost_shipping, labels = c("No", "Yes"))
  df_known$form_of_address <- factor(df_known$form_of_address, labels = c("Company","Mr","Mrs"))
  df_known$model <- factor(df_known$model) 
  df_known$goods_value <- factor(df_known$goods_value)
  df_known$cost_shipping <- factor(df_known$cost_shipping)
  df_known$referrer <- factor(df_known$referrer)
  df_known$return_customer <- factor(df_known$return_customer, labels = c("No", "Yes"))
  df_known$advertising_code <- mapvalues(df_known$advertising_code, from = c(""), to = c("NoCode"))
  return(df_known) 
  }else if(data=="class"){
    df_class<- read.csv2( "assignment_BADS_WS1617_class.csv", header = TRUE, sep = ",", dec= ":"  )
    df_class$order_date <- ymd(df_class$order_date) 
    df_class$account_creation_date<- ymd(df_class$account_creation_date)
    df_class$deliverydate_actual <- ymd(df_class$deliverydate_actual) #NA values !
    df_class$deliverydate_estimated <- ymd(df_class$deliverydate_estimated)
    df_class$title <- factor(df_class$title, labels = c("No","Yes"))
    df_class$giftwrapping   <- factor(df_class$giftwrapping,labels =c("No", "Yes"))
    df_class$coupon <- factor(df_class$coupon, labels = c("No", "Yes"))
    df_class$newsletter <- factor(df_class$newsletter, labels = c("No", "Yes"))
    df_class$delivery <- factor(df_class$delivery, labels = c("No", "Yes"))
    df_class$referrer <- factor(df_class$referrer, labels = c("No", "Yes"))
    df_class$cost_shipping <- factor(df_class$cost_shipping, labels = c("No", "Yes"))
    df_class$form_of_address <- factor(df_class$form_of_address, labels = c("Company","Mr","Mrs"))
    df_class$model <- factor(df_class$model) 
    df_class$goods_value <- factor(df_class$goods_value)
    df_class$cost_shipping <- factor(df_class$cost_shipping)
    df_class$referrer <- factor(df_class$referrer)
    df_class$advertising_code <- mapvalues(df_class$advertising_code, from = c(""), to = c("NoCode"))
    return(df_class) 
  }
}  

### Date differences ###

create.diff.delivery <- function(x){
  x$diff_delivery<-as.Date(x$deliverydate_estimated, format="%Y/%m/%d")-
    as.Date(x$deliverydate_actual, format="%Y/%m/%d")
  return(x)
}
create.diff.registration <- function(x){
  x$diff_registration_order<-as.Date(x$order_date, format="%Y/%m/%d")-
    as.Date(x$account_creation_date, format="%Y/%m/%d")
  return(x)
}
create.deliverytime.estimated <- function(x){
  x$deliverytime_estimated<-as.numeric(as.Date(x$deliverydate_estimated, format="%Y/%m/%d")-
    as.Date(x$order_date, format="%Y/%m/%d"))
  return(x)
}
create.deliverytime.actual <- function(x){
  x$deliverytime_actual<-as.numeric(as.Date(x$deliverydate_actual, format="%Y/%m/%d")-
    as.Date(x$order_date, format="%Y/%m/%d"))
  return(x)
}


evaluation.costMatrix1 <- function(yhat,y){
  # this function calculate, given a prob prediction vector and a class 
  # validation vector the maximum monetary gain we can achieve given the costMatrix
  # for each tao and return the maximum 
  
  tao <- seq(from = 0.05, to = 1, by = 0.001 )  
  results <- vector(mode = "numeric",length = length(tao))
  for ( i in 1:length(tao)){
    yhat.class <- ifelse(yhat >= tao[i], 1, 0)
    cm <- table(yhat.class,y)
    #cm <- confusionMatrix(yhat.class, y.validation)
    results[i] <- cm[1,1]*3+(-cm[1,2]*10)
    
  }
  
  max_gain <- max(results)
  tao_max <- tao[which(results %in% max(results))]
  nn.return <- 1-(sum(y)/length(y))
  max_potential_gain <- length(y)*nn.return*3
  percentage_potential <- max_gain/max_potential_gain
  
  # This last line rapresent an importanto figure: the pecentage of obtained gains 
  # with respect of the maximum gain achievable 
  # good way to evaluate all the models we do whith no concern of the lenght of the partition
  #that we take
  
  
  eval.cm <- list("max_gain" = max_gain,"best_tao" =tao_max, 
                  "max_potential_gain"= max_potential_gain,
                  "percentage_potential" = percentage_potential)

  return(eval.cm)
}

standardize1 <- function(x){
  
  my <- mean(x)
  std <- sd(x)
  result <- (x-my)/std
  return(result)
}

post_code_areas <- function(df_known = NULL ){
  # This function reduce the levels of the variable postcode invoice by dividing them in the
  # different geographical areas following the german postcode distribution:
  # (https://en.wikipedia.org/wiki/List_of_postal_codes_in_Germany)
  
  for ( i in 1:nrow(df_known)){
    df_known$postcode_invoice[i]<- floor(df_known$postcode_invoice[i]/10) 
    
  }
  
  df_known$postcode_invoice  <- as.factor(df_known$postcode_invoice)
  return(df_known)}



day_week_FE <- function(my_df = NULL){
  # This function creates a new variable which rapresent the day of the week in which the order was placed 
  # It was build taking as a reference the 2013 calendar and extracting from the comparison between it and the
  # order date the corrisponding day of the week
  first <- ymd("2013-01-01")
  day_week <- vector(mode = "character", length = nrow(my_df))
  for ( i in 1:nrow(my_df)){
    dist <- as.numeric(my_df$order_date[i] - first)+1
    sett <-  floor(dist/7)
    giorno <- dist - sett*7
    if (giorno == 1){
      day_week[i] <- "Tusday"
    }
    else if ( giorno == 2){
      day_week[i] <- "Wednesday"
    }
    else if ( giorno == 3){
      day_week[i] <- "Thursday"
    }
    else if ( giorno == 4){
      day_week[i] <- "Friday"
    }
    else if ( giorno == 5){
      day_week[i] <- "Saturday"
    }
    else if ( giorno == 6){
      day_week[i] <- "Sunday"
    }
    else if ( giorno == 0){
      day_week[i] <- "Monday"
    }
    
  }
  my_df$day_week <- as.factor(day_week)
  return(my_df)}



Features_engineering <- function(my_df = NULL){
  # This function cointains different feature engineering processes. It creates a coloumn which tells us 
  # if the order was palced less than 20 days before christams, 
  # a coloumn which identify orders that request giftwrapping and that were remitted ( we consider them as wrong presents)
  # and finally a binary coloumn which  identify if the order was placed either friday, saturday or sunday.
  
  
  xmas <- as.Date("2013/12/24", format="%Y/%m/%d")
  # Create a coloumn for xmans buy
  xmas_buy <-  vector(mode = "numeric", length = nrow(my_df))
  for ( i in 1: nrow(my_df)){
    diff <- my_df$order_date[i] - xmas
    if(diff > -20 & diff < 0){
      xmas_buy[i] <- 1}
    else{xmas_buy[i] <- 0}
    
  }
  my_df$xmas_buy <- as.factor(xmas_buy)
  
  
  
  
  # Wrong present
  wrong_present <- vector(mode = "numeric", length = nrow(my_df))
  for ( i in 1:nrow(my_df)){
    if (my_df$giftwrapping[i] == "Yes" & my_df$remitted_items[i] != "No Item"){
      wrong_present[i] <- 1
    }
    else{wrong_present[i]<- 0}
  } #
  
  
  my_df$wrong_present <- as.factor(wrong_present)
  
  
  
  
  ## Week end order
  weekend_ord <- vector(mode = "numeric", length = length(my_df))
  for ( i in 1:nrow(my_df)){
    if(is.na(my_df$day_week[i])){weekend_ord[i] <- 0}
    else if (my_df$day_week[i] == "Saturday" | my_df$day_week[i] == "Sunday" ){weekend_ord[i] <- 1}
    
    else {weekend_ord[i] <- 0}
  }
  my_df$weekend_order <- as.factor(weekend_ord)
  
  
  return(my_df)}