library(rstan)
library(dplyr)


data <- read.csv("../data/clean/state_clean.csv")
# Since running simulation on the entire dataset requires
# too much computational power, limit with 10k rows only
df <- data[sample(nrow(data), 10000), ]
df$state_id <- as.integer(factor(df$state))
state_levels <- levels(factor(df$state))
n_states <- n_distinct(df$state_id)
df_clean <- df[!is.na(df$coverage_per_state), ]
observed <- df_clean[!is.na(df_clean$total_patients_hospitalized_confirmed_influenza_and_covid), ]
missing  <- df_clean[is.na(df_clean$total_patients_hospitalized_confirmed_influenza_and_covid), ]

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
  vector[N_states] alpha_raw;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N_states] alpha;
  alpha = mu_alpha + tau_alpha * alpha_raw;
}
model {
  // Priors
  mu_alpha ~ normal(0, 50);
  tau_alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sigma ~ normal(0, 5);
  alpha_raw ~ normal(0, 1);
  vector[N_obs] mu;
  mu = alpha[state_id] + beta * x;

  # Enforce non-negativity by using truncated normal distribution
  # lccdf is log complementary CDF of the norm dist
  target += normal_lpdf(y | mu, sigma)
            - normal_lccdf(0 | mu, sigma);
}
"

dat <- list(N_obs = nrow(observed),N_states = n_states,
  y = observed$total_patients_hospitalized_confirmed_influenza_and_covid,
  x = observed$coverage_per_state,state_id = observed$state_id)

# using multithreading, all cores on my laptop (4)
pp_post <- rstan::stan(model_code = pp_mod,data = dat,chains = 4,iter=2000,cores=4)

# Extract posteriors
post <- rstan::extract(pp_post)
alpha_means <- colMeans(post$alpha)
beta_mean   <- mean(post$beta)

to_impute <- data #df_clean
missing_idx <- is.na(to_impute$total_patients_hospitalized_confirmed_influenza_and_covid)
pred <- alpha_means[to_impute$state_id[missing_idx]] + 
  beta_mean * to_impute$coverage_per_state[missing_idx]
to_impute$total_patients_hospitalized_confirmed_influenza_and_covid[missing_idx] <- 
  pmax(pred, 0)

# Save all coefficients separately so that we don't have to run the
# entire model again
state_lookup <- unique(df_clean[, c("state_id", "state")])
state_lookup <- state_lookup[order(state_lookup$state_id), ]
coef_df <- data.frame(state=state_lookup$state,state_id=state_lookup$state_id,alpha= alpha_means,
  beta= beta_mean)

write.csv(coef_df, "../data/clean/imputation_coefficients.csv", row.names = FALSE)
# # lazy way to drop a column
to_impute$state_id <- NULL
write.csv(to_impute, "../data/clean/state_imputed.csv", row.names = FALSE)
