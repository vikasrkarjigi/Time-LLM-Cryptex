df = read.csv("dataset/cryptex/train_data/ret-h-14d_trains.csv")
df = df[df["State"] == "COMPLETE",]
df = df[, !(names(df) %in% c("State", "Param.batch_size", "Number", "UserAttribute.dataset", "UserAttribute.granularity", "UserAttribute.target"))]
head(df)

model = lm(Value ~ ., data = df)
summary(model)

x_bar = mean(df$Value)
n = length(df$Value)
p_value = pt(t_stat, n - 1, lower.tail = FALSE)
t_stat = (x_bar - .5) / (sd(df$Value) / sqrt(length(df$Value)))

hist(df$Value, breaks = 20)

