
library(rstan)
library(dplyr)


sample <- read.csv("data/clean/sample.csv")
sample$state_id <- as.integer(factor(sample$state))
n_states <- n_distinct(sample$state_id)

observed <- sample[!is.na(sample$previous_day_total_ed_visits_7_day_sum), ]
missing  <- sample[is.na(sample$previous_day_total_ed_visits_7_day_sum), ]

pp_mod <- "
data {
  int N_obs;
  int N_states;
  vector[N_obs] y;
  vector[N_obs] x;
  array[N_obs] int state_id;
}
parameters {
  real mu_alpha;
  real<lower=0> tau_alpha;
  vector[N_states] alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  mu_alpha ~ normal(-21000, 2000);
  tau_alpha ~ normal(0, 2000);
  sigma ~ normal(0, 2000);
  beta~ normal(27000, 5000);
  alpha~ normal(mu_alpha, tau_alpha);
  y ~ normal(alpha[state_id] + beta * x, sigma);
}
"

dat <- list(N_obs= nrow(observed),N_states = n_states,y= observed$previous_day_total_ed_visits_7_day_sum,
  x= observed$inpatient_beds_utilization,state_id = observed$state_id)

fit <- rstan::stan(model_code = pp_mod, data = dat, chains = 4, iter = 2000)
alpha_means <- colMeans(rstan::extract(fit, pars = "alpha")$alpha)
beta_mean   <- mean(rstan::extract(fit, pars = "beta")$beta)
state_levels <- levels(factor(sample$state))

# load and prep the data
to_impute <- read.csv("data/clean/sample_to_impute.csv")
to_impute$state_id <- match(to_impute$state, state_levels)

# impute using coefficients from simulation
missing_idx <- is.na(to_impute$previous_day_total_ed_visits_7_day_sum)
to_impute$previous_day_total_ed_visits_7_day_sum[missing_idx] <- 
  alpha_means[to_impute$state_id[missing_idx]] + 
  beta_mean * to_impute$inpatient_beds_utilization[missing_idx]

to_impute$state_id <- NULL # lazy way to drop state id
write.csv(to_impute, "data/clean/imputed.csv", row.names = FALSE)