library(tidyverse)
library(glmnet)

unfiltered_df <- as_tibble(read.csv("dataset/cryptex/params_analysis/llama3.1_ret-h-4m_params.csv", header = TRUE))
df = unfiltered_df %>% 
  filter(State == "COMPLETE") %>% 
  filter("Param.pred_len" != 2) %>% 
  select(-c(State, `Param.batch_size`, Number, 
                   `UserAttribute.dataset`, 
                   `UserAttribute.granularity`, 
                   `UserAttribute.target`, 
                   `Param.pred_len`))
        
X <- model.matrix(Value ~ . , data = df)[, -1, drop = FALSE]

# 2) Response as numeric vector
y <- df$Value
if (is.factor(y)) y <- as.numeric(y)  # or better: as.numeric(y) - 1 for 0/1

# 3) Deal with NA/NaN/Inf in X or y
keep <- is.finite(y) & is.finite(rowSums(X))  # drops any row with non-finite
X <- X[keep, , drop = FALSE]
y <- y[keep]


# (Optional) median-impute any remaining NAs columnwise (shouldn't be needed after keep)
for (j in seq_len(ncol(X))) {
  if (anyNA(X[, j])) X[is.na(X[, j]), j] <- median(X[, j], na.rm = TRUE)
}

# 4) Drop zero-variance columns (glmnet can handle, but safer to clean)
nzv <- apply(X, 2, function(col) sd(col, na.rm = TRUE) > 0)
X <- X[, nzv, drop = FALSE]

# 5) CV Lasso
cvfit <- cv.glmnet(X, y, alpha = 1, family = "gaussian")  # use family="binomial" for 0/1
cvfit$lambda.min
cvfit$lambda.1se

coef(cvfit, s = "lambda.min")


x_bar <- mean(df$Value)
n <- length(df$Value)
t_stat <- (x_bar - .5) / (sd(df$Value) / sqrt(length(df$Value)))
p_value <- pt(t_stat, n - 1, lower.tail = FALSE)

# Print t-statistic
# This indicates whether the parameter is significantly different from 0.5
print(t_stat)

hist(df$Value, breaks = 20)
