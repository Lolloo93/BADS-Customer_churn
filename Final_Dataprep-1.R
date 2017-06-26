

source("help_functions.R")
library("plyr")
library("ggplot2")
if(!require("klaR")) install.packages("klaR"); library("klaR")
if(!require("Information")) install.packages("Information"); library("Information")
if(!require("DMwR")) install.packages("DMwR"); library("DMwR")
if(!require("FNN")) install.packages("FNN"); library("FNN")
if(!require("rpart")) install.packages("rpart"); library("rpart")
if(!require("rpart.plot")) install.packages("rpart.plot"); library("rpart.plot")
if(!require("caret")) install.packages("caret"); library("caret")
if(!require("pROC")) install.packages("pROC"); library("pROC")
if(!require("randomForest")) install.packages("randomForest"); library("randomForest")
if(!require("foreach")) install.packages("foreach"); library("foreach")
if(!require("doParallel")) install.packages("doParallel"); library("doParallel") # load the package
if(!require("microbenchmark")) install.packages("microbenchmark"); library("microbenchmark")

nrofcores <- detectCores()
cl <- makeCluster(max(detectCores()-1,1))
registerDoParallel(cl)
message(paste("\n Registered number of cores:\n",getDoParWorkers(),"\n"))



df_known <- get.data("known")

### 1.) discretize weight ###

df_known$weightnew[is.na(df_known$weight)] <- "Missing Value"
df_known$weightnew[df_known$weight==0] <- "No weight"
df_known$weightnew[df_known$weight>0 & df_known$weight<=500] <- "Low weight"
df_known$weightnew[df_known$weight>500 & df_known$weight<=1000] <- "Medium weight"
df_known$weightnew[df_known$weight>1000] <- "Heavy weight"

df_known$weightnew<- as.factor(df_known$weightnew)


### 2.) remove points redeemed, ID ###

df_known <- df_known[,-c(1,19)]

### 3.) Calculate time differences ###

df_known<- create.diff.delivery(df_known)
df_known<- create.diff.registration(df_known)
df_known<- create.deliverytime.actual(df_known) 
df_known<- create.deliverytime.estimated(df_known) 

### 4.) Create additional binary "advertising code" variable

df_known$advertising_code_binary <- as.character(df_known$advertising_code)

for (i in 1:length(df_known$advertising_code_binary)) {
  if(df_known$advertising_code_binary[i] == "NoCode"){
    df_known$advertising_code_binary[i] <- "No"
  }else{
    df_known$advertising_code_binary[i] <- "Yes"
  }
}
df_known$advertising_code_binary <- as.factor(df_known$advertising_code_binary)
df_known$advertising_code <- as.factor(df_known$advertising_code)

### 5.) Transform order date and account creation date to month-year 

df_known$order_date_new <- format(as.Date(df_known$order_date), "%Y-%m")
df_known$account_creation_date_new <- format(as.Date(df_known$account_creation_date), "%Y-%m")

### 6.) Remove old account date and old weight variable ###

df_known <- df_known[,-c(5,21)]

### 7.) 

df_known$diff_registration_order[is.na(df_known$diff_registration_order)] <- 0

for(i in 1:length(df_known$account_creation_date_new)){
  if(is.na(df_known$account_creation_date_new[i])){
    df_known$account_creation_date_new[i] <- df_known$order_date_new[i]
  }
}

# 8.) Treat NA's in form of adress as "Missing" #

df_known$form_of_address <- factor(x= df_known$form_of_address, levels = c("Mr", "Mrs", "Company", "Missing"))
df_known$form_of_address[is.na(df_known$form_of_address)] <- "Missing"

# 9.) Handle NA's and outliers for the deliverytime variables #

# 9.1) Deliverytime_estimated has 55 hugely negative entries,all because delivery_date_estimated is in 2010 #
#      Treat them as technical errors and set them as NA #

df_known$deliverytime_estimated[df_known$deliverytime_estimated <0] <- -1

# 9.2) Same for the extreme huge values 

df_known$deliverytime_estimated[df_known$deliverytime_estimated > 371] <- 372


# 9.3) Deliverytime_estimated not so extreme values:

# - estimated delivery times have big jump to interval [367,371] where around 2910 cases are in
# --> set these dates all to 365 to "mark" them

# - if there is no actual delivery date it is mostly no physical goods, canceled items or no goods at all
# --> set these delivery times to 0 as it makes sense

df_known$deliverytime_estimated[df_known$deliverytime_estimated > 365 & df_known$deliverytime_estimated <372] <- 365 
df_known$deliverytime_actual[is.na(df_known$deliverytime_actual)] <- 0

# 9.4) Recalculate diff_delivery #

for (i in 1:length(df_known$diff_delivery)) {
  df_known$diff_delivery[i] <- df_known$deliverytime_actual[i] - df_known$deliverytime_estimated[i]
}

# 9.5) Remove delivery_date_actual and delivery_date_estimated #
# Since their important information are now incorporated through the above steps and they are not needed anymore

df_known <- df_known[,-c(18,19)]

# Final Check if there are any NA's left in the data set #

for (Var in names(df_known)) {
  missing <- sum(is.na(df_known[,Var]))
  if (missing > 0) {
    print(c(Var,missing))
  }
}
################################################################################################################

############ Feature Engineering ############

### Discretize the count variables  ###

# backup
df_known2 <- df_known

# index vector for loop:

idx.count_var <- c(14, 18:31)

for (i in idx.count_var) {
  quantile0.99 <- quantile(df_known2[,i], probs = 0.99)
  
  for (j in 1:length(df_known2$return_customer)) {
    if(df_known2[j,i] == 0 && df_known2[j,i] <= quantile0.99){
      df_known2[j,i] <- 0
    }else if(df_known2[j,i] == 1 && df_known2[j,i] <= quantile0.99){
      df_known2[j,i] <- 1
    }else if(df_known2[j,i] >=2 && df_known2[j,i] <= quantile0.99){
      df_known2[j,i] <- 2
    } else if(df_known2[j,i] >=2 && df_known2[j,i] > quantile0.99){
      df_known2[j,i] <- 3
    }
  }
  df_known2[,i] <- factor(df_known2[,i], levels = c(0,1,2,3), labels = c("No Item", "One Item", "Multiple Items", "Extreme")) 
}

### use of help functions to generate some new features (see description in help_functions.R) ###

df_known2 <- day_week_FE(df_known2)

df_known2 <- post_code_areas(df_known2)

df_known2 <- Features_engineering(df_known2)

### remove post_code_delivery and old order date ###

df_known2 <- df_known2[,-c(1,10)]

### Standardize Remaining Numeric Variables ###

df_known2$diff_delivery <- as.numeric(df_known2$diff_delivery)
df_known2$diff_registration_order <- as.numeric(df_known2$diff_registration_order)

idx.numeric <- sapply(df_known2, is.numeric)
df_known2[,idx.numeric] <- lapply(df_known2[,idx.numeric], standardize1)

###  Create interaction variable between referrer and coupon ###

df_known2$referrer_coupon <- factor(x=0, levels = c(0,1,2,3),labels = c("Referrer and Coupon","Referrer and no Coupon",
                                                                        "Coupon and no Referrer","No Coupon and no Referrer"))


df_known2$referrer_coupon[df_known2$referrer=="No" & df_known2$coupon=="Yes"] <- "Coupon and no Referrer"
df_known2$referrer_coupon[df_known2$referrer=="Yes" & df_known2$coupon=="No"] <- "Referrer and no Coupon"
df_known2$referrer_coupon[df_known2$referrer=="No" & df_known2$coupon=="No"] <- "No Coupon and no Referrer"


###  Use WOE for transforming order_date_new and creation_date_new, e-mail, postcode_order, referrer_coupon, advertising code and weightnew (as defined by my arbitrary binning) ###

if(!require("klaR")) install.packages("klaR"); library("klaR")
if(!require("Information")) install.packages("Information"); library("Information")


# Dates as factor: 

df_known2$order_date_new <- as.factor(df_known2$order_date_new)
df_known2$account_creation_date_new <- as.factor(df_known2$account_creation_date_new)

# Create train set to estimate WOE and IV:

woe.idx.train <- createDataPartition(y = df_known2$return_customer, p = 0.5, list = FALSE) # Draw a random, stratified sample including p percent of the data
woe.train <- df_known2[woe.idx.train,] 


# Estimate WOE object using klaR package

woe.object1 <- woe(return_customer ~ email_domain + postcode_invoice + referrer_coupon + advertising_code + order_date_new + account_creation_date_new + weightnew , data = woe.train, zeroadj = 0.5)
woe.object1$woe
woe.object1$IV # according to IV, e-mail adress and postcode should be deleted, the rest can stay in although only weakly predictive

# Add estimates to train.rf1 set:

df_known3 <- predict(woe.object1, newdata = df_known2, replace = TRUE)

# Remove woe.email_domain and woe.postcode_invoice due to low IV

df_known3 <- df_known3[,-c(37,38)]


### 2.) Test to see if delivery is linear dependent from payment

if(!require("gmodels")) install.packages("gmodels"); library("gmodels")

with(df_known3, CrossTable(payment, delivery))

# table confirms: delivery is "yes" if and only if payment is "cash"
# --> delivery can be removed

df_known3 <- df_known3[,-6]


##################################################################################################################



