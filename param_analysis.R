library(tidyverse)

unfiltered_df <- as_tibble(read.csv("dataset/cryptex/params_analysis/llama3.1_ret-h-4m_params.csv", header = TRUE))
df = unfiltered_df %>% 
  filter(State == "COMPLETE") %>% 
  filter("Param.pred_len" == 2) %>% 
  select(-c(State, `Param.batch_size`, Number, 
                   `UserAttribute.dataset`, 
                   `UserAttribute.granularity`, 
                   `UserAttribute.target`, 
                   `Param.pred_len`))
         

model <- lm(Value ~ ., data = filtered_df)
summary(model)

x_bar <- mean(df$Value)
n <- length(df$Value)
t_stat <- (x_bar - .5) / (sd(df$Value) / sqrt(length(df$Value)))
p_value <- pt(t_stat, n - 1, lower.tail = FALSE)

# Print t-statistic
# This indicates whether the parameter is significantly different from 0.5
print(t_stat)

hist(df$Value, breaks = 20)
