library(tidyverse)
library(glmnet)
unfiltered_df1 = as_tibble(read.csv("dataset/cryptex/params_analysis/llama3.1_ret-h-4m-20-epochs_params.csv", header = TRUE))
unfiltered_df2 <- as_tibble(read.csv("dataset/cryptex/params_analysis/llama3.1_ret-h-4m_params.csv", header = TRUE))


unfiltered_df = rbind(unfiltered_df1, unfiltered_df2)

unfiltered_df = as_tibble(read.csv("dataset/cryptex/params_analysis/llama3.1_ohlcv-h-4m-new-params_params.csv", header = TRUE))

df = unfiltered_df %>% 
  filter(State == "COMPLETE") %>% 
  filter(Value > .0001) %>% 
  select(-c(State, Number, 
                   `UserAttribute.dataset`, 
                   `UserAttribute.granularity`, 
                   `UserAttribute.target`, 
                   `Param.pred_len`))


names(df)
df <- df[vapply(df, function(x) length(unique(x)) > 1, logical(1))]
c(df[1,])
# Basic Stats
model = lm(Value ~ .,data = df)
summary(model)
df
x_bar <- mean(df$Value)
n <- length(df$Value)
t_stat <- (x_bar - .5) / (sd(df$Value) / sqrt(n))
p_value <- pt(t_stat, n - 1, lower.tail = FALSE)

# Print t-statistic
# This indicates whether the parameter is significantly different from 0.5
t_stat
p_value

hist(df$Value, breaks = 20)

sd(df$Value)
mean(df$Value)

t.test(df$Value, mu = .5)
