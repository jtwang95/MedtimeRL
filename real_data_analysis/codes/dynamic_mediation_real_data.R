library(tidyverse)
library(geepack)
library(mgcv)
library(parallel)

clusbootglm_sample_glm <- function(f, i, Obsno, model, family, data, p, res.or) {
  j <- f[, i]
  obs <- unlist(Obsno[j])
  coef <- rep(NA, p) # added
  bootcoef <- tryCatch(coef(glm(model, family = family, data = data[obs, ])),
    warning = function(x) rep(as.numeric(NA), p)
  )
  ifelse(length(bootcoef) == p, coef <- as.vector(bootcoef), coef[which(names(res.or$coef) %in% names(bootcoef))] <- bootcoef)
  return(coef)
}

my_clusbootglm <- function(model, data, clusterid, family = gaussian, B = 5000,
                           confint.level = 0.95, n.cores = 1) {
  tt_cores <- detectCores()
  if (n.cores > tt_cores) {
    message(sprintf(
      "Note: \"n.cores\" was set to %d, but only %d are available. Using all cores.",
      n.cores, tt_cores
    ))
  }
  model <- as.formula(model)
  res.or <- glm(model, family = family, data = data)
  confint.pboundaries <- c((1 - confint.level) / 2, 1 - (1 - confint.level) / 2)
  confint.Zboundaries <- qnorm(confint.pboundaries)
  n <- nrow(data)
  p <- length(res.or$coef)
  coefs <- matrix(NA, nrow = B, ncol = p)
  arguments <- as.list(match.call())
  clusterid <- eval(arguments$clusterid, data)
  cluster <- as.character(clusterid)
  clusters <- unique(cluster)
  Obsno <- split(1:n, cluster)
  f <- matrix(clusters, length(clusters), B)
  ff <- matrix(f, prod(dim(f)), 1)
  fff <- sample(ff)
  f <- matrix(fff, length(clusters), B)
  if (is.numeric(n.cores) & n.cores > 0) {
    if (n.cores == 1) {
      for (i in 1:B) {
        j <- f[, i]
        obs <- unlist(Obsno[j])
        bootcoef <- tryCatch(coef(glm(model,
          family = family,
          data = data[obs, ]
        )), warning = function(x) {
          rep(
            as.numeric(NA),
            p
          )
        })
        coefs[i, which(names(res.or$coef) %in% names(bootcoef))] <- bootcoef
      }
    }
    if (n.cores > 1) {
      cl <- makeCluster(max(min(n.cores, tt_cores), 2))
      previous_RNGkind <- RNGkind()[1]
      RNGkind("L'Ecuyer-CMRG")
      nextRNGStream(.Random.seed)
      clusterExport(cl,
        varlist = c(
          "f", "Obsno",
          "model", "family", "data",
          "p", "res.or", "clusbootglm_sample_glm"
        ),
        envir = environment()
      )
      splitclusters <- 1:B
      out <- parSapplyLB(cl, splitclusters, function(x) {
        clusbootglm_sample_glm(
          f,
          x, Obsno, model, family, data, p, res.or
        )
      })
      coefs <- t(out)
      stopCluster(cl)
      RNGkind(previous_RNGkind)
    }
  }
  invalid.samples <- colSums(is.na(coefs))
  names(invalid.samples) <- colnames(coefs) <- names(res.or$coef)
  samples.with.NA.coef <- which(is.na(rowSums(coefs)))
  sdcoefs <- apply(coefs, 2, sd, na.rm = TRUE)
  result <- list(
    call = match.call(), model = model, family = family,
    B = B, coefficients = coefs, data = data, bootstrap.matrix = f,
    subject.vector = clusterid, lm.coefs = res.or$coef, boot.coefs = colMeans(coefs,
      na.rm = TRUE
    ), boot.sds = sdcoefs, ci.level = confint.level
  )
  class(result) <- "clusbootglm"
  return(result)
}


#####################################################################################################################################################################

load(
  "~/Dropbox (University of Michigan)/ihs_data/2018/imputed_data/2018_imputed_data_daily_hr.RData"
)

dat <- impute_list[[9]]$full_data_complete %>%
  select(-c(NOTIFICATION_TYPE, date_day)) %>%
  mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
  group_by(USERID, study_week, week_category, specialty) %>%
  summarise(across(everything(), ~ mean(.x))) %>%
  ungroup() %>%
  mutate(msg_sent = 0 + (week_category != "None")) %>%
  left_join(impute_list[[1]]$all_baseline)

##########################################
tmp <- dat %>% mutate(msg_sent = (week_category == "activity") + 0)

formula_backbone <- "~ study_week * msg_sent + Sex + PHQtot0 + Neu0 + depr0 + EFE0 + pre_intern_step_count + pre_intern_sleep_count + pre_intern_resting_hr + pre_intern_rmssd"
all_mediators <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, "+", paste(all_mediators[-i], collapse = " + "), sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")

  result <- my_clusbootglm(f,
    data = tmp, family = gaussian(link = "identity"), clusterid = USERID, B = 1000, n.cores = 8
  )
  v <- result$coefficients
  res0 <- t(rbind(
    result$lm.coefs["msg_sent"] + 0:25 * result$lm.coefs["study_week:msg_sent"],
    apply(replicate(26, v[, "msg_sent"]) + sapply(0:25, function(i) i * v[, "study_week:msg_sent"]),
      2,
      FUN = function(x) quantile(x, c(0.025, 0.975))
    )
  )) %>% data.frame()
  colnames(res0) <- c("effect", "effect_lb", "effect_ub")
  res0 <- res0 %>%
    tibble() %>%
    mutate(week = 0:25, response = rep(mediator, 26))
  res <- res %>% bind_rows(res0)
}

p1 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of activity msg") +
  facet_wrap(~response, scales = "free", ncol = 4)
p11 <- ggplot(data = res %>% filter(response %in% c("rmssd")), aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of activity message") +
  facet_wrap(~response, scales = "free", ncol = 1, strip.position = "right", labeller = labeller(response = c("rmssd" = "HRV", "SLEEP_COUNT" = "Sleep")))

##########################################
tmp <- dat %>% mutate(msg_sent = (week_category == "sleep") + 0)

formula_backbone <- "~ study_week * msg_sent + Sex + PHQtot0 + Neu0 + depr0 + EFE0 + pre_intern_step_count + pre_intern_sleep_count + pre_intern_resting_hr + pre_intern_rmssd"
all_mediators <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, "+", paste(all_mediators[-i], collapse = " + "), sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")

  result <- my_clusbootglm(f,
    data = tmp, family = gaussian(link = "identity"), clusterid = USERID, B = 1000, n.cores = 8
  )
  v <- result$coefficients
  res0 <- t(rbind(
    result$lm.coefs["msg_sent"] + 0:25 * result$lm.coefs["study_week:msg_sent"],
    apply(replicate(26, v[, "msg_sent"]) + sapply(0:25, function(i) i * v[, "study_week:msg_sent"]),
      2,
      FUN = function(x) quantile(x, c(0.025, 0.975))
    )
  )) %>% data.frame()
  colnames(res0) <- c("effect", "effect_lb", "effect_ub")
  res0 <- res0 %>%
    tibble() %>%
    mutate(week = 0:25, response = rep(mediator, 26))
  res <- res %>% bind_rows(res0)
}

p2 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of sleep msg") +
  facet_wrap(~response, scales = "free", ncol = 4)

p22 <- ggplot(data = res %>% filter(response %in% c("SLEEP_COUNT")), aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of sleep message") +
  facet_wrap(~response, scales = "free", ncol = 1, strip.position = "right", labeller = labeller(response = c("rmssd" = "HRV", "SLEEP_COUNT" = "Sleep")))

##########################################
tmp <- dat %>% mutate(msg_sent = (week_category == "mood") + 0)

formula_backbone <- "~ study_week * msg_sent + Sex + PHQtot0 + Neu0 + depr0 + EFE0 + pre_intern_step_count + pre_intern_sleep_count + pre_intern_resting_hr + pre_intern_rmssd"
all_mediators <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, "+", paste(all_mediators[-i], collapse = " + "), sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")

  result <- my_clusbootglm(f,
    data = tmp, family = gaussian(link = "identity"), clusterid = USERID, B = 1000, n.cores = 8
  )
  v <- result$coefficients
  res0 <- t(rbind(
    result$lm.coefs["msg_sent"] + 0:25 * result$lm.coefs["study_week:msg_sent"],
    apply(replicate(26, v[, "msg_sent"]) + sapply(0:25, function(i) i * v[, "study_week:msg_sent"]),
      2,
      FUN = function(x) quantile(x, c(0.025, 0.975))
    )
  )) %>% data.frame()
  colnames(res0) <- c("effect", "effect_lb", "effect_ub")
  res0 <- res0 %>%
    tibble() %>%
    mutate(week = 0:25, response = rep(mediator, 26))
  res <- res %>% bind_rows(res0)
}

p3 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of mood msg") +
  facet_wrap(~response, scales = "free", ncol = 4)
p33 <- ggplot(data = res %>% filter(response %in% c("SLEEP_COUNT")), aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of mood message") +
  facet_wrap(~response, scales = "free", ncol = 1, strip.position = "right", labeller = labeller(response = c("rmssd" = "HRV", "SLEEP_COUNT" = "Sleep")))

##########################################
tmp <- dat %>% mutate(msg_sent = (week_category != "None") + 0)

formula_backbone <- "~ study_week * msg_sent + Sex + PHQtot0 + Neu0 + depr0 + EFE0 + pre_intern_step_count + pre_intern_sleep_count + pre_intern_resting_hr + pre_intern_rmssd"
all_mediators <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd", "MOOD")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, "+", paste(all_mediators[-i], collapse = " + "), sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")

  result <- my_clusbootglm(f,
    data = tmp, family = gaussian(link = "identity"), clusterid = USERID, B = 1000, n.cores = 8
  )
  v <- result$coefficients
  res0 <- t(rbind(
    result$lm.coefs["msg_sent"] + 0:25 * result$lm.coefs["study_week:msg_sent"],
    apply(replicate(26, v[, "msg_sent"]) + sapply(0:25, function(i) i * v[, "study_week:msg_sent"]),
      2,
      FUN = function(x) quantile(x, c(0.025, 0.975))
    )
  )) %>% data.frame()
  colnames(res0) <- c("effect", "effect_lb", "effect_ub")
  res0 <- res0 %>%
    tibble() %>%
    mutate(week = 0:25, response = rep(mediator, 26))
  res <- res %>% bind_rows(res0)
}

p4 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of msg") +
  facet_wrap(~response, scales = "free", ncol = 4)
p44 <- ggplot(data = res %>% filter(response %in% c("rmssd")), aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of message") +
  facet_wrap(~response, scales = "free", ncol = 1, strip.position = "right", labeller = labeller(response = c("rmssd" = "HRV", "SLEEP_COUNT" = "Sleep")))

##########################################
tmp <- dat %>% mutate(msg_sent = (week_category != "None") + 0)

formula_backbone <- "~ study_week * msg_sent + Sex + PHQtot0 + Neu0 + depr0 + EFE0 + pre_intern_step_count + pre_intern_sleep_count + pre_intern_resting_hr + pre_intern_rmssd + STEP_COUNT + SLEEP_COUNT+resting_hr+rmssd"
all_mediators <- c("MOOD")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")

  result <- my_clusbootglm(f,
    data = tmp, family = gaussian(link = "identity"), clusterid = USERID, B = 1000, n.cores = 8
  )
  v <- result$coefficients
  res0 <- t(rbind(
    result$lm.coefs["msg_sent"] + 0:25 * result$lm.coefs["study_week:msg_sent"],
    apply(replicate(26, v[, "msg_sent"]) + sapply(0:25, function(i) i * v[, "study_week:msg_sent"]),
      2,
      FUN = function(x) quantile(x, c(0.025, 0.975))
    )
  )) %>% data.frame()
  colnames(res0) <- c("effect", "effect_lb", "effect_ub")
  res0 <- res0 %>%
    tibble() %>%
    mutate(week = 0:25, response = rep(mediator, 26))
  res <- res %>% bind_rows(res0)
}

p5 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of message on MOOD")
p5

fit.gam <- gam(
  resting_hr ~ msg_sent + s(study_week, by = factor(msg_sent), bs = "ps", k = 4) + Sex + PHQtot0 + Neu0 + depr0 + EFE0 + pre_intern_step_count + pre_intern_sleep_count + pre_intern_resting_hr + pre_intern_rmssd + STEP_COUNT + SLEEP_COUNT + rmssd,
  data = tmp,
  method = "REML"
)
# knots = list(study_week=c(6,13,20)),

pdat <- expand.grid(
  study_week = seq(0, 25),
  msg_sent = c(0, 1),
  Sex = 1,
  PHQtot0 = mean(tmp$PHQtot0),
  Neu0 = mean(tmp$Neu0),
  depr0 = mean(tmp$depr0),
  EFE0 = mean(tmp$EFE0),
  pre_intern_step_count = mean(tmp$pre_intern_step_count),
  pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
  pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
  pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
  STEP_COUNT = mean(tmp$STEP_COUNT),
  SLEEP_COUNT = mean(tmp$SLEEP_COUNT),
  rmssd = mean(tmp$rmssd)
)
xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
c1 <- grepl("msg", colnames(xp))
r1 <- with(pdat, msg_sent == 1)
r0 <- with(pdat, msg_sent == 0)
X <- xp[r1, ] - xp[r0, ]
X[, !c1] <- 0
dif <- X %*% coef(fit.gam)
se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
crit <- qt(.975, df.residual(fit.gam))
upr <- dif + (crit * se)
lwr <- dif - (crit * se)

p51 <- ggplot(
  data.frame(
    week = seq(0, 25),
    effect = dif,
    effect_lb = lwr,
    effect_ub = upr
  ),
  aes(x = week, y = effect)
) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(
    yintercept = 0,
    color = "red",
    linetype = "dashed"
  ) +
  xlab("Week") +
  ylab("Effect of msg on weekly average mood score")
p51

p52 <- tmp %>%
  group_by(study_week, msg_sent) %>%
  summarize(mean_mood = mean(MOOD)) %>%
  pivot_wider(values_from = mean_mood, names_from = msg_sent, names_prefix = "mean_mood") %>%
  mutate(diff = mean_mood1 - mean_mood0) %>%
  ggplot(aes(x = study_week, y = diff)) +
  theme_bw() +
  geom_line() +
  ylab("Unadjusted difference of mood score")

ggarrange(p5, p51, p52, ncol = 3)

##########################################
##########################################
dat <- impute_list[[1]]$full_data_complete %>%
  select(-c(NOTIFICATION_TYPE, date_day)) %>%
  mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
  group_by(USERID, study_week, week_category, specialty) %>%
  summarise(across(everything(), ~ mean(.x))) %>%
  ungroup() %>%
  left_join(impute_list[[1]]$all_baseline)
tmp <- dat %>%
  mutate(msg_sent = (week_category != "None") + 0) %>%
  arrange(USERID, study_week) %>%
  group_by(USERID) %>%
  mutate(
    prev_MOOD = lag(MOOD, 1),
    prev_STEP_COUNT = lag(STEP_COUNT, 1),
    prev_SLEEP_COUNT = lag(SLEEP_COUNT, 1),
    prev_resting_hr = lag(resting_hr, 1),
    prev_rmssd = lag(rmssd, 1)
  )
tmp <- tmp %>%
  select(
    USERID, pre_intern_mood, pre_intern_sleep_count,
    pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
  ) %>%
  unique() %>%
  rename(
    prev_mood_ = "pre_intern_mood",
    prev_step_count_ = "pre_intern_step_count",
    prev_sleep_count_ = "pre_intern_sleep_count",
    prev_resting_hr_ = "pre_intern_resting_hr",
    prev_rmssd_ = "pre_intern_rmssd"
  ) %>%
  mutate(study_week = 1) %>%
  right_join(tmp) %>%
  mutate(
    prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
    prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
    prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
    prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
    prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
  )

common_control_vars <- paste(c(
  "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
  "pre_intern_sleep_count", "pre_intern_resting_hr",
  "pre_intern_rmssd", "pre_intern_mood"
), collapse = "+")

formula_backbone <- paste("~ study_week * msg_sent", common_control_vars, sep = "+")
all_mediators <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, "+", paste("prev", mediator, sep = "_"), sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")
  mu_est <- unname(sapply(
    0:25,
    FUN = function(i) {
      fit$coefficients["msg_sent"] + i * fit$coefficients["study_week:msg_sent"]
    }
  ))
  vcov_matrix <- vcov(fit)
  std_est <- unname(sapply(
    0:25,
    FUN = function(i) {
      sqrt(vcov_matrix["msg_sent", "msg_sent"] + i**2 * vcov_matrix["study_week:msg_sent", "study_week:msg_sent"] + 2 *
        i * vcov_matrix["msg_sent", "study_week:msg_sent"])
    }
  ))
  res0 <- data.frame(
    week = 0:25,
    effect = mu_est,
    effect_lb = mu_est - 1.96 * std_est,
    effect_ub = mu_est + 1.96 * std_est,
    response = rep(mediator, 26)
  )
  res <- res %>% bind_rows(res0)
}


f <- paste("MOOD", formula_backbone, "+", paste("prev", "MOOD", sep = "_"), sep = " ")
f <- as.formula(f)
fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")
mu_est <- unname(sapply(
  0:25,
  FUN = function(i) {
    fit$coefficients["msg_sent"] + i * fit$coefficients["study_week:msg_sent"]
  }
))
vcov_matrix <- vcov(fit)
std_est <- unname(sapply(
  0:25,
  FUN = function(i) {
    sqrt(vcov_matrix["msg_sent", "msg_sent"] + i**2 * vcov_matrix["study_week:msg_sent", "study_week:msg_sent"] + 2 *
      i * vcov_matrix["msg_sent", "study_week:msg_sent"])
  }
))
res0 <- data.frame(
  week = 0:25,
  effect = mu_est,
  effect_lb = mu_est - 1.96 * std_est,
  effect_ub = mu_est + 1.96 * std_est,
  response = rep("mood", 26)
)
res <- res %>% bind_rows(res0)

f <- paste("MOOD ~ msg_sent + s(study_week, by = factor(msg_sent),bs='ps',k=4)", common_control_vars, "prev_MOOD", sep = "+")
fit.gam <- gam(
  as.formula(f),
  data = tmp,
  method = "REML"
)
# knots = list(study_week=c(6,13,20)),

pdat <- expand.grid(
  study_week = seq(0, 25),
  msg_sent = c(0, 1),
  Sex = 1,
  PHQtot0 = mean(tmp$PHQtot0),
  Neu0 = mean(tmp$Neu0),
  depr0 = mean(tmp$depr0),
  EFE0 = mean(tmp$EFE0),
  pre_intern_step_count = mean(tmp$pre_intern_step_count),
  pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
  pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
  pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
  pre_intern_mood = mean(tmp$pre_intern_mood),
  STEP_COUNT = mean(tmp$STEP_COUNT),
  SLEEP_COUNT = mean(tmp$SLEEP_COUNT),
  resting_hr = mean(tmp$resting_hr),
  rmssd = mean(tmp$rmssd),
  prev_MOOD = mean(tmp$prev_MOOD)
)
xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
c1 <- grepl("msg", colnames(xp))
r1 <- with(pdat, msg_sent == 1)
r0 <- with(pdat, msg_sent == 0)
X <- xp[r1, ] - xp[r0, ]
X[, !c1] <- 0
dif <- X %*% coef(fit.gam)
se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
crit <- qt(.975, df.residual(fit.gam))
upr <- dif + (crit * se)
lwr <- dif - (crit * se)
res0 <- data.frame(effect = dif, effect_lb = lwr, effect_ub = upr, week = 0:25, response = "mood_smooth") %>% tibble()
res <- res %>% bind_rows(res0)

res <- res %>% mutate(
  ymin = case_when(
    response == "rmssd" ~ -1.5,
    response == "SLEEP_COUNT" ~ -0.07,
    response == "STEP_COUNT" ~ -0.15,
    response == "resting_hr" ~ -0.3,
    response == "mood" ~ -0.12,
    response == "mood_smooth" ~ -0.125
  ),
  ymax = case_when(
    response == "rmssd" ~ 3.65,
    response == "SLEEP_COUNT" ~ 0.082,
    response == "STEP_COUNT" ~ 0.16,
    response == "resting_hr" ~ 0.6,
    response == "mood" ~ 0.1,
    response == "mood_smooth" ~ 0.145
  )
)

p6 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 6, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "mood" = "Mood", "mood_smooth" = "Mood (smoothed)"
    ))
  ) +
  geom_blank(aes(y = ymin)) +
  geom_blank(aes(y = ymax))

#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 1),
      prev_STEP_COUNT = lag(STEP_COUNT, 1),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 1),
      prev_resting_hr = lag(resting_hr, 1),
      prev_rmssd = lag(rmssd, 1)
    )
  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 1) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ study_week * msg_sent", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    f <- paste(v, formula_backbone, "+", paste("prev", v, sep = "_"), sep = " ")
    f <- as.formula(f)
    fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits, mi = TRUE) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  if (mi == TRUE) {
    require("mitools")
    require("miceadds")
    cmod <- MIextract(fits, fun = coef)
    vmod <- MIextract(fits, fun = vcov)
    res <- miceadds::pool_mi(qhat = cmod, u = vmod)
  } else {
    res <- fits
  }
  res.coef <- coef(res)
  res.vcov <- vcov(res)

  var_name <- "msg_sent"
  interaction_term <- "study_week:msg_sent"
  p_est <- unname(sapply(
    1:26,
    FUN = function(i) {
      res.coef[var_name] + i * res.coef[interaction_term]
    }
  ))
  std_est <- unname(sapply(
    1:26,
    FUN = function(i) {
      sqrt(res.vcov[var_name, var_name] + i**2 * res.vcov[interaction_term, interaction_term] + 2 *
        i * res.vcov[var_name, interaction_term])
    }
  ))
  res <- data.frame(
    week = 1:26,
    effect = p_est,
    effect_lb = p_est - 1.96 * std_est,
    effect_ub = p_est + 1.96 * std_est
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 1:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 1:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 1:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 1:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 1:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}

# res = res %>% mutate(ymin=case_when(response == "rmssd" ~ -1.5,
#                                     response == "SLEEP_COUNT" ~ -0.07,
#                                     response == "STEP_COUNT" ~ -0.15,
#                                     response == "resting_hr" ~ -0.3,
#                                     response == "mood" ~ -0.12),
#                      ymax=case_when(response == "rmssd" ~ 3.65,
#                                     response == "SLEEP_COUNT" ~ 0.082,
#                                     response == "STEP_COUNT" ~ 0.16,
#                                     response == "resting_hr" ~ 0.6,
#                                     response == "mood" ~ 0.1))

p7 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
# geom_blank(aes(y = ymin)) +
# geom_blank(aes(y = ymax))

#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 2),
      prev_STEP_COUNT = lag(STEP_COUNT, 2),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 2),
      prev_resting_hr = lag(resting_hr, 2),
      prev_rmssd = lag(rmssd, 2),
      msg_sent_prev = lag(msg_sent, 1),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    ) %>%
    filter(study_week >= 2)

  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 2) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd", "MOOD"
  ), collapse = "+")

  formula_backbone <- paste("~ study_week*msg_sent_prev", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    f <- paste(v, formula_backbone, "+", paste("prev", v, sep = "_"), sep = " ")
    f <- as.formula(f)
    fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits, mi = TRUE) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  if (mi == TRUE) {
    require("mitools")
    require("miceadds")
    cmod <- MIextract(fits, fun = coef)
    vmod <- MIextract(fits, fun = vcov)
    res <- miceadds::pool_mi(qhat = cmod, u = vmod)
  } else {
    res <- fits
  }
  res.coef <- coef(res)
  res.vcov <- vcov(res)

  var_name <- "msg_sent_prev"
  interaction_term <- "study_week:msg_sent_prev"
  p_est <- unname(sapply(
    2:26,
    FUN = function(i) {
      res.coef[var_name] + i * res.coef[interaction_term]
    }
  ))
  std_est <- unname(sapply(
    2:26,
    FUN = function(i) {
      sqrt(res.vcov[var_name, var_name] + i**2 * res.vcov[interaction_term, interaction_term] + 2 *
        i * res.vcov[var_name, interaction_term])
    }
  ))
  res <- data.frame(
    week = 2:26,
    effect = p_est,
    effect_lb = p_est - 1.96 * std_est,
    effect_ub = p_est + 1.96 * std_est
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 2:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 2:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 2:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 2:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 2:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}


p8 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
p8
# geom_blank(aes(y = ymin)) +
# geom_blank(aes(y = ymax))

#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 3),
      prev_STEP_COUNT = lag(STEP_COUNT, 3),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 3),
      prev_resting_hr = lag(resting_hr, 3),
      prev_rmssd = lag(rmssd, 3),
      msg_sent_prev_lag1 = lag(msg_sent, 1),
      msg_sent_prev_lag2 = lag(msg_sent, 2),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    ) %>%
    filter(study_week >= 3)

  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 3) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ study_week * msg_sent_prev_lag1 + study_week * msg_sent_prev_lag2", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    # f = paste(v,formula_backbone,"+",paste("prev",v,sep="_"),sep=" ")
    f <- paste(v, formula_backbone, sep = " ")
    f <- as.formula(f)
    fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits, mi = TRUE) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  if (mi == TRUE) {
    require("mitools")
    require("miceadds")
    cmod <- MIextract(fits, fun = coef)
    vmod <- MIextract(fits, fun = vcov)
    res <- miceadds::pool_mi(qhat = cmod, u = vmod)
  } else {
    res <- fits
  }
  res.coef <- coef(res)
  res.vcov <- vcov(res)

  var_name1 <- "msg_sent_prev_lag1"
  interaction_term1 <- "study_week:msg_sent_prev_lag1"
  var_name2 <- "msg_sent_prev_lag2"
  interaction_term2 <- "study_week:msg_sent_prev_lag2"
  p_est <- unname(sapply(
    3:26,
    FUN = function(i) {
      res.coef[var_name1] + res.coef[var_name2] + i * (res.coef[interaction_term1] + res.coef[interaction_term2])
    }
  ))
  std_fun <- function(i) {
    v <- res.vcov[var_name1, var_name1] + res.vcov[var_name2, var_name2]
    v <- v + i**2 * (res.vcov[interaction_term1, interaction_term1] + res.vcov[interaction_term2, interaction_term2])
    cv <- 2 * i * (res.vcov[var_name1, interaction_term1] + res.vcov[var_name1, interaction_term2])
    cv <- cv + 2 * i * (res.vcov[var_name2, interaction_term1] + res.vcov[var_name2, interaction_term2])
    cv <- cv + 2 * res.vcov[var_name1, var_name2]
    cv <- cv + 2 * i**2 * res.vcov[interaction_term1, interaction_term2]
    return(sqrt(v + cv))
  }
  std_est <- unname(sapply(
    3:26,
    FUN = std_fun
  ))
  res <- data.frame(
    week = 3:26,
    effect = p_est,
    effect_lb = p_est - 1.96 * std_est,
    effect_ub = p_est + 1.96 * std_est
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 3:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 3:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 3:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 3:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 3:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}

# res = res %>% mutate(ymin=case_when(response == "rmssd" ~ -1.5,
#                                     response == "SLEEP_COUNT" ~ -0.07,
#                                     response == "STEP_COUNT" ~ -0.15,
#                                     response == "resting_hr" ~ -0.3,
#                                     response == "mood" ~ -0.12),
#                      ymax=case_when(response == "rmssd" ~ 3.65,
#                                     response == "SLEEP_COUNT" ~ 0.082,
#                                     response == "STEP_COUNT" ~ 0.16,
#                                     response == "resting_hr" ~ 0.6,
#                                     response == "mood" ~ 0.1))

p9 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
p9
# geom_blank(aes(y = ymin)) +
# geom_blank(aes(y = ymax))

#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 2),
      prev_STEP_COUNT = lag(STEP_COUNT, 2),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 2),
      prev_resting_hr = lag(resting_hr, 2),
      prev_rmssd = lag(rmssd, 2),
      msg_sent_prev = lag(msg_sent, 1),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    ) %>%
    filter(study_week >= 2)

  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 2) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent_prev + s(study_week, by = factor(msg_sent_prev),bs='ps',k=5)", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    # f = paste(v,formula_backbone,"+",paste("prev",v,sep="_"),sep=" ")
    f <- paste(v, formula_backbone, sep = " ")
    f <- as.formula(f)
    fit <- gam(f, data = tmp)
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  res <- tibble()
  for (i in 1:length(fits)) {
    fit.gam <- fits[[i]]
    pdat <- expand.grid(
      study_week = seq(2, 26),
      msg_sent_prev = c(0, 1),
      Sex = 1,
      PHQtot0 = mean(tmp$PHQtot0),
      Neu0 = mean(tmp$Neu0),
      depr0 = mean(tmp$depr0),
      EFE0 = mean(tmp$EFE0),
      pre_intern_step_count = mean(tmp$pre_intern_step_count),
      pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
      pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
      pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
      pre_intern_mood = mean(tmp$pre_intern_mood),
      prev_STEP_COUNT = mean(tmp$prev_STEP_COUNT),
      prev_SLEEP_COUNT = mean(tmp$prev_SLEEP_COUNT),
      prev_resting_hr = mean(tmp$prev_resting_hr),
      prev_rmssd = mean(tmp$prev_rmssd),
      MOOD = mean(tmp$MOOD)
    )
    xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
    c1 <- grepl("msg", colnames(xp))
    r1 <- with(pdat, msg_sent_prev == 1)
    r0 <- with(pdat, msg_sent_prev == 0)
    X <- xp[r1, ] - xp[r0, ]
    X[, !c1] <- 0
    dif <- X %*% coef(fit.gam)
    se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
    crit <- qt(.975, df.residual(fit.gam))
    upr <- dif + (crit * se)
    lwr <- dif - (crit * se)
    res0 <- data.frame(effect = dif, effect_lb = lwr, effect_ub = upr, week = 2:26, mi_idx = i) %>% tibble()
    res <- res %>% bind_rows(res0)
  }

  res_agg <- res %>%
    group_by(week) %>%
    summarise(across(where(is.numeric), ~ mean(.x)))

  res <- data.frame(
    week = 2:26,
    effect = res_agg$effect,
    effect_lb = res_agg$effect_lb,
    effect_ub = res_agg$effect_ub
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 2:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 2:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 2:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 2:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 2:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}

# res = res %>% mutate(ymin=case_when(response == "rmssd" ~ -1.5,
#                                     response == "SLEEP_COUNT" ~ -0.07,
#                                     response == "STEP_COUNT" ~ -0.15,
#                                     response == "resting_hr" ~ -0.3,
#                                     response == "mood" ~ -0.12),
#                      ymax=case_when(response == "rmssd" ~ 3.65,
#                                     response == "SLEEP_COUNT" ~ 0.082,
#                                     response == "STEP_COUNT" ~ 0.16,
#                                     response == "resting_hr" ~ 0.6,
#                                     response == "mood" ~ 0.1))

p10 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Delayed effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
# geom_blank(aes(y = ymin)) +
# geom_blank(aes(y = ymax))
p10

#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 3),
      prev_STEP_COUNT = lag(STEP_COUNT, 3),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 3),
      prev_resting_hr = lag(resting_hr, 3),
      prev_rmssd = lag(rmssd, 3),
      msg_sent_prev_lag1 = lag(msg_sent, 1),
      msg_sent_prev_lag2 = lag(msg_sent, 2)
    ) %>%
    filter(study_week >= 3)


  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 3) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
    )

  tmp$rmssd <- log(tmp$rmssd)
  tmp$pre_intern_rmssd <- log(tmp$pre_intern_rmssd)
  tmp$prev_rmssd <- tmp$prev_rmssd

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "pre_intern_mood"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent_prev_lag1 + msg_sent_prev_lag2 + s(study_week, by = factor(msg_sent_prev_lag1),bs='ps',k=5) + s(study_week, by = factor(msg_sent_prev_lag2),bs='ps',k=5)", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    f <- paste(v, formula_backbone, "+", paste("prev", v, sep = "_"), sep = " ")
    f <- as.formula(f)
    fit <- gam(f, data = tmp)
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits, mi = TRUE) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  if (mi == TRUE) {
    require("mitools")
    require("miceadds")
    cmod <- MIextract(fits, fun = coef)
    vmod <- MIextract(fits, fun = vcov)
    res <- miceadds::pool_mi(qhat = cmod, u = vmod)
  } else {
    res <- fits
  }
  res.coef <- coef(res)
  res.vcov <- vcov(res)

  var_name1 <- "msg_sent_prev_lag1"
  interaction_term1 <- "study_week:msg_sent_prev_lag1"
  var_name2 <- "msg_sent_prev_lag2"
  interaction_term2 <- "study_week:msg_sent_prev_lag2"
  p_est <- unname(sapply(
    3:26,
    FUN = function(i) {
      res.coef[var_name1] + res.coef[var_name2] + i * (res.coef[interaction_term1] + res.coef[interaction_term2])
    }
  ))
  std_fun <- function(i) {
    v <- res.vcov[var_name1, var_name1] + res.vcov[var_name2, var_name2]
    v <- v + i**2 * (res.vcov[interaction_term1, interaction_term1] + res.vcov[interaction_term2, interaction_term2])
    cv <- 2 * i * (res.vcov[var_name1, interaction_term1] + res.vcov[var_name1, interaction_term2])
    cv <- cv + 2 * i * (res.vcov[var_name2, interaction_term1] + res.vcov[var_name2, interaction_term2])
    cv <- cv + 2 * res.vcov[var_name1, var_name2]
    cv <- cv + 2 * i**2 * res.vcov[interaction_term1, interaction_term2]
    return(sqrt(v + cv))
  }
  std_est <- unname(sapply(
    3:26,
    FUN = std_fun
  ))
  res <- data.frame(
    week = 3:26,
    effect = p_est,
    effect_lb = p_est - 1.96 * std_est,
    effect_ub = p_est + 1.96 * std_est
  )
  res
}

fun <- function(fits) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  res <- tibble()
  for (i in 1:length(fits)) {
    fit.gam <- fits[[i]]
    pdat <- expand.grid(
      study_week = seq(3, 26),
      msg_sent_prev_lag1 = c(0, 1),
      msg_sent_prev_lag2 = c(0, 1),
      Sex = 1,
      PHQtot0 = mean(tmp$PHQtot0),
      Neu0 = mean(tmp$Neu0),
      depr0 = mean(tmp$depr0),
      EFE0 = mean(tmp$EFE0),
      pre_intern_step_count = mean(tmp$pre_intern_step_count),
      pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
      pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
      pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
      pre_intern_mood = mean(tmp$pre_intern_mood),
      prev_STEP_COUNT = mean(tmp$prev_STEP_COUNT),
      prev_SLEEP_COUNT = mean(tmp$prev_SLEEP_COUNT),
      prev_resting_hr = mean(tmp$prev_resting_hr),
      prev_rmssd = mean(tmp$prev_rmssd)
    )
    xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
    c1 <- grepl("msg", colnames(xp))
    r1 <- with(pdat, msg_sent_prev_lag1 == 1 & msg_sent_prev_lag2 == 1)
    r0 <- with(pdat, msg_sent_prev_lag1 == 0 & msg_sent_prev_lag2 == 0)
    X <- xp[r1, ] - xp[r0, ]
    X[, !c1] <- 0
    dif <- X %*% coef(fit.gam)
    se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
    crit <- qt(.975, df.residual(fit.gam))
    upr <- dif + (crit * se)
    lwr <- dif - (crit * se)
    res0 <- data.frame(effect = dif, effect_lb = lwr, effect_ub = upr, week = 3:26, mi_idx = i) %>% tibble()
    res <- res %>% bind_rows(res0)
  }

  res_agg <- res %>%
    group_by(week) %>%
    summarise(across(where(is.numeric), ~ mean(.x)))

  res <- data.frame(
    week = 3:26,
    effect = res_agg$effect,
    effect_lb = res_agg$effect_lb,
    effect_ub = res_agg$effect_ub
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 3:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 3:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 3:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 3:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 3:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}
res <- res %>%
  arrange(response, week) %>%
  group_by(response) %>%
  mutate(effect_cumsum = cummean(effect))
# res = res %>% mutate(ymin=case_when(response == "rmssd" ~ -1.5,
#                                     response == "SLEEP_COUNT" ~ -0.07,
#                                     response == "STEP_COUNT" ~ -0.15,
#                                     response == "resting_hr" ~ -0.3,
#                                     response == "mood" ~ -0.12),
#                      ymax=case_when(response == "rmssd" ~ 3.65,
#                                     response == "SLEEP_COUNT" ~ 0.082,
#                                     response == "STEP_COUNT" ~ 0.16,
#                                     response == "resting_hr" ~ 0.6,
#                                     response == "mood" ~ 0.1))

p11 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
p11

#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 1),
      prev_STEP_COUNT = lag(STEP_COUNT, 1),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 1),
      prev_resting_hr = lag(resting_hr, 1),
      prev_rmssd = lag(rmssd, 1)
    )
  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 1) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent+s(study_week, by = factor(msg_sent),bs='ps',k=5)", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    f <- paste(v, formula_backbone, "+", paste("prev", v, sep = "_"), sep = " ")
    f <- as.formula(f)
    fit <- gam(f, data = tmp)
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  res <- tibble()
  for (i in 1:length(fits)) {
    fit.gam <- fits[[i]]
    pdat <- expand.grid(
      study_week = seq(1, 26),
      msg_sent = c(0, 1),
      Sex = 1,
      PHQtot0 = mean(tmp$PHQtot0),
      Neu0 = mean(tmp$Neu0),
      depr0 = mean(tmp$depr0),
      EFE0 = mean(tmp$EFE0),
      pre_intern_step_count = mean(tmp$pre_intern_step_count),
      pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
      pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
      pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
      pre_intern_mood = mean(tmp$pre_intern_mood),
      prev_STEP_COUNT = mean(tmp$prev_STEP_COUNT),
      prev_SLEEP_COUNT = mean(tmp$prev_SLEEP_COUNT),
      prev_resting_hr = mean(tmp$prev_resting_hr),
      prev_rmssd = mean(tmp$prev_rmssd)
    )
    xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
    c1 <- grepl("msg", colnames(xp))
    r1 <- with(pdat, msg_sent == 1)
    r0 <- with(pdat, msg_sent == 0)
    X <- xp[r1, ] - xp[r0, ]
    X[, !c1] <- 0
    dif <- X %*% coef(fit.gam)
    se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
    crit <- qt(.975, df.residual(fit.gam))
    upr <- dif + (crit * se)
    lwr <- dif - (crit * se)
    res0 <- data.frame(effect = dif, effect_lb = lwr, effect_ub = upr, week = 1:26, mi_idx = i) %>% tibble()
    res <- res %>% bind_rows(res0)
  }

  res_agg <- res %>%
    group_by(week) %>%
    summarise(across(where(is.numeric), ~ mean(.x)))

  res <- data.frame(
    week = 1:26,
    effect = res_agg$effect,
    effect_lb = res_agg$effect_lb,
    effect_ub = res_agg$effect_ub
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 1:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 1:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 1:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 1:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 1:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}


p12 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Immediate effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
p12

###########################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 2),
      prev_STEP_COUNT = lag(STEP_COUNT, 2),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 2),
      prev_resting_hr = lag(resting_hr, 2),
      prev_rmssd = lag(rmssd, 2),
      msg_sent_prev = lag(msg_sent, 1),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    ) %>%
    filter(study_week >= 2)

  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 2) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent_prev + s(study_week, by = factor(msg_sent_prev),bs='ps',k=5)", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    # f = paste(v,formula_backbone,"+",paste("prev",v,sep="_"),sep=" ")
    f <- paste(v, formula_backbone, sep = " ")
    f <- as.formula(f)
    fit <- gam(f, data = tmp)
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  res <- tibble()
  for (i in 1:length(fits)) {
    fit.gam <- fits[[i]]
    pdat <- expand.grid(
      study_week = seq(2, 26),
      msg_sent_prev = c(0, 1),
      Sex = 1,
      PHQtot0 = mean(tmp$PHQtot0),
      Neu0 = mean(tmp$Neu0),
      depr0 = mean(tmp$depr0),
      EFE0 = mean(tmp$EFE0),
      pre_intern_step_count = mean(tmp$pre_intern_step_count),
      pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
      pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
      pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
      pre_intern_mood = mean(tmp$pre_intern_mood),
      prev_STEP_COUNT = mean(tmp$prev_STEP_COUNT),
      prev_SLEEP_COUNT = mean(tmp$prev_SLEEP_COUNT),
      prev_resting_hr = mean(tmp$prev_resting_hr),
      prev_rmssd = mean(tmp$prev_rmssd),
      MOOD = mean(tmp$MOOD)
    )
    xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
    c1 <- grepl("msg", colnames(xp))
    r1 <- with(pdat, msg_sent_prev == 1)
    r0 <- with(pdat, msg_sent_prev == 0)
    X <- xp[r1, ] - xp[r0, ]
    X[, !c1] <- 0
    dif <- X %*% coef(fit.gam)
    se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
    crit <- qt(.975, df.residual(fit.gam))
    upr <- dif + (crit * se)
    lwr <- dif - (crit * se)
    res0 <- data.frame(effect = dif, effect_lb = lwr, effect_ub = upr, week = 2:26, mi_idx = i) %>% tibble()
    res <- res %>% bind_rows(res0)
  }

  res_agg <- res %>%
    group_by(week) %>%
    summarise(across(where(is.numeric), ~ mean(.x)))

  res <- data.frame(
    week = 2:26,
    effect = res_agg$effect,
    effect_lb = res_agg$effect_lb,
    effect_ub = res_agg$effect_ub
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 2:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 2:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 2:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 2:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 2:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}

res <- res %>%
  arrange(response, week) %>%
  group_by(response) %>%
  mutate(effect_cumsum = cummean(effect))

p13 <- ggplot(data = res, aes(x = week, y = effect_cumsum)) +
  theme_bw() +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Delayed cumulative effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
# geom_blank(aes(y = ymin)) +
# geom_blank(aes(y = ymax))
p13
#######################################################################################################################

# MI
fits.cluster_step <- list()
fits.cluster_sleep <- list()
fits.cluster_mood <- list()
fits.cluster_rhr <- list()
fits.cluster_hrv <- list()

results <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 1),
      prev_STEP_COUNT = lag(STEP_COUNT, 1),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 1),
      prev_resting_hr = lag(resting_hr, 1),
      prev_rmssd = lag(rmssd, 1)
    )
  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 1) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent+s(study_week, by = factor(msg_sent),bs='ps',k=5)", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    f <- paste(v, formula_backbone, "+", paste("prev", v, sep = "_"), sep = " ")
    f <- as.formula(f)
    fit <- gam(f, data = tmp)
    if (v == "STEP_COUNT") {
      fits.cluster_step[[i]] <- fit
    } else if (v == "SLEEP_COUNT") {
      fits.cluster_sleep[[i]] <- fit
    } else if (v == "resting_hr") {
      fits.cluster_rhr[[i]] <- fit
    } else if (v == "rmssd") {
      fits.cluster_hrv[[i]] <- fit
    } else if (v == "MOOD") {
      fits.cluster_mood[[i]] <- fit
    }
  }
}

fun <- function(fits) {
  # https://www.rdocumentation.org/packages/miceadds/versions/3.11-6/topics/pool_mi
  res <- tibble()
  for (i in 1:length(fits)) {
    fit.gam <- fits[[i]]
    pdat <- expand.grid(
      study_week = seq(1, 26),
      msg_sent = c(0, 1),
      Sex = 1,
      PHQtot0 = mean(tmp$PHQtot0),
      Neu0 = mean(tmp$Neu0),
      depr0 = mean(tmp$depr0),
      EFE0 = mean(tmp$EFE0),
      pre_intern_step_count = mean(tmp$pre_intern_step_count),
      pre_intern_sleep_count = mean(tmp$pre_intern_sleep_count),
      pre_intern_resting_hr = mean(tmp$pre_intern_resting_hr),
      pre_intern_rmssd = mean(tmp$pre_intern_rmssd),
      pre_intern_mood = mean(tmp$pre_intern_mood),
      prev_STEP_COUNT = mean(tmp$prev_STEP_COUNT),
      prev_SLEEP_COUNT = mean(tmp$prev_SLEEP_COUNT),
      prev_resting_hr = mean(tmp$prev_resting_hr),
      prev_rmssd = mean(tmp$prev_rmssd)
    )
    xp <- predict(fit.gam, newdata = pdat, type = "lpmatrix")
    c1 <- grepl("msg", colnames(xp))
    r1 <- with(pdat, msg_sent == 1)
    r0 <- with(pdat, msg_sent == 0)
    X <- xp[r1, ] - xp[r0, ]
    X[, !c1] <- 0
    dif <- X %*% coef(fit.gam)
    se <- sqrt(diag(X %*% vcov(fit.gam) %*% t(X)))
    crit <- qt(.975, df.residual(fit.gam))
    upr <- dif + (crit * se)
    lwr <- dif - (crit * se)
    res0 <- data.frame(effect = dif, effect_lb = lwr, effect_ub = upr, week = 1:26, mi_idx = i) %>% tibble()
    res <- res %>% bind_rows(res0)
  }

  res_agg <- res %>%
    group_by(week) %>%
    summarise(across(where(is.numeric), ~ mean(.x)))

  res <- data.frame(
    week = 1:26,
    effect = res_agg$effect,
    effect_lb = res_agg$effect_lb,
    effect_ub = res_agg$effect_ub
  )
  res
}

res <- tibble()
for (j in 1:length(all_vars)) {
  v <- all_vars[j]
  print(v)
  if (v == "STEP_COUNT") {
    res0 <- fun(fits = fits.cluster_step) %>% mutate(week = 1:26, response = "STEP_COUNT")
  } else if (v == "SLEEP_COUNT") {
    res0 <- fun(fits = fits.cluster_sleep) %>% mutate(week = 1:26, response = "SLEEP_COUNT")
  } else if (v == "resting_hr") {
    res0 <- fun(fits = fits.cluster_rhr) %>% mutate(week = 1:26, response = "resting_hr")
  } else if (v == "rmssd") {
    res0 <- fun(fits = fits.cluster_hrv) %>% mutate(week = 1:26, response = "rmssd")
  } else if (v == "MOOD") {
    res0 <- fun(fits = fits.cluster_mood) %>% mutate(week = 1:26, response = "MOOD")
  }
  res <- res %>% bind_rows(res0)
}


p14 <- ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Immediate effect of push notifications") +
  facet_wrap(~response,
    scales = "free_y", ncol = 5, strip.position = "top",
    labeller = labeller(response = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )
p14

#######################################################################################################################

fun <- function(.x, .y) {
  fit <- gam(effect ~ ns(week, df = 5), data = .x)
  effect_smooth <- predict(fit, newdata = data.frame(week = .x$week))
  .x %>% mutate(effect_smooth = effect_smooth)
}

# MI
# fits.cluster_step = list()
# fits.cluster_sleep = list()
# fits.cluster_mood = list()
# fits.cluster_rhr = list()
# fits.cluster_hrv = list()

results1 <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 1),
      prev_STEP_COUNT = lag(STEP_COUNT, 1),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 1),
      prev_resting_hr = lag(resting_hr, 1),
      prev_rmssd = lag(rmssd, 1)
    )
  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 1) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    f <- paste(v, formula_backbone, "+", paste("prev", v, sep = "_"), sep = " ")
    f <- as.formula(f)
    for (k in unique(tmp$study_week)) {
      tmp1 <- tmp %>% filter(study_week == k)
      fit <- lm(f, data = tmp1)
      results1 <- results1 %>% bind_rows(data.frame(est = fit$coefficients["msg_sent"], week = k, imp = i, mediator = v))
    }
  }
}

p16_immediate <- results1 %>%
  group_by(mediator, week) %>%
  summarize(effect = mean(est)) %>%
  group_by(mediator) %>%
  group_modify(fun) %>%
  ggplot(aes(x = week, y = effect_smooth)) +
  theme_bw() +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  xlab("Week") +
  ylab("Immediate effect of push notifications") +
  facet_wrap(~mediator,
    scales = "free_y", ncol = 4, strip.position = "top",
    labeller = labeller(mediator = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  ) + 
  theme(axis.title.y = element_blank())
p16_immediate

results2 <- tibble()
for (i in 1:impute_list$num_impute) {
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 2),
      prev_STEP_COUNT = lag(STEP_COUNT, 2),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 2),
      prev_resting_hr = lag(resting_hr, 2),
      prev_rmssd = lag(rmssd, 2),
      msg_sent_prev = lag(msg_sent, 1),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    ) %>%
    filter(study_week >= 2)

  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 2) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd", "prev_STEP_COUNT", "prev_SLEEP_COUNT", "prev_resting_hr", "prev_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent_prev", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    # f = paste(v,formula_backbone,"+",paste("prev",v,sep="_"),sep=" ")
    f <- paste(v, formula_backbone, sep = " ")
    f <- as.formula(f)
    for (k in unique(tmp$study_week)) {
      tmp1 <- tmp %>% filter(study_week == k)
      fit <- lm(f, data = tmp1)
      results2 <- results2 %>% bind_rows(data.frame(est = fit$coefficients["msg_sent_prev"], week = k, imp = i, mediator = v))
    }
  }
}

p16_delayed <- results2 %>%
  arrange(mediator, week) %>%
  group_by(mediator, imp) %>%
  mutate(est_cummean = cummean(est)) %>%
  group_by(mediator, week) %>%
  summarize(effect = mean(est_cummean)) %>%
  group_by(mediator) %>%
  group_modify(fun) %>%
  ggplot(aes(x = week, y = effect_smooth)) +
  theme_bw() +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  xlab("Week") +
  ylab("Delayed effect of push notifications") +
  facet_wrap(~mediator,
    scales = "free_y", ncol = 4, strip.position = "top",
    labeller = labeller(mediator = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )+ 
  theme(axis.title.y = element_blank())
p16_delayed

###########################################################################################
library(foreach)
library(doParallel)
fun <- function(.x, .y) {
  fit <- gam(effect ~ ns(week, df = 4), data = .x)
  effect_smooth <- predict(fit, newdata = data.frame(week = .x$week))
  .x %>% mutate(effect_smooth = effect_smooth)
}

registerDoParallel(8)
results3 <- (foreach(i = 1:20, .combine = rbind) %dopar% {
  res <- tibble()
  dat <- impute_list[[i]]$full_data_complete %>%
    select(-c(NOTIFICATION_TYPE, date_day)) %>%
    mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
    group_by(USERID, study_week, week_category, specialty) %>%
    summarise(across(everything(), ~ mean(.x))) %>%
    ungroup() %>%
    left_join(impute_list[[i]]$all_baseline)
  tmp <- dat %>%
    mutate(msg_sent = (week_category != "None") + 0) %>%
    arrange(USERID, study_week) %>%
    group_by(USERID) %>%
    mutate(
      prev_MOOD = lag(MOOD, 1),
      prev_STEP_COUNT = lag(STEP_COUNT, 1),
      prev_SLEEP_COUNT = lag(SLEEP_COUNT, 1),
      prev_resting_hr = lag(resting_hr, 1),
      prev_rmssd = lag(rmssd, 1)
    )
  tmp <- tmp %>%
    select(
      USERID, pre_intern_mood, pre_intern_sleep_count,
      pre_intern_step_count, pre_intern_resting_hr, pre_intern_rmssd
    ) %>%
    unique() %>%
    rename(
      prev_mood_ = "pre_intern_mood",
      prev_step_count_ = "pre_intern_step_count",
      prev_sleep_count_ = "pre_intern_sleep_count",
      prev_resting_hr_ = "pre_intern_resting_hr",
      prev_rmssd_ = "pre_intern_rmssd"
    ) %>%
    mutate(study_week = 1) %>%
    right_join(tmp) %>%
    mutate(
      prev_MOOD = if_else(is.na(prev_MOOD), prev_mood_, prev_MOOD),
      prev_STEP_COUNT = if_else(is.na(prev_STEP_COUNT), prev_step_count_, prev_STEP_COUNT),
      prev_SLEEP_COUNT = if_else(is.na(prev_SLEEP_COUNT), prev_sleep_count_, prev_SLEEP_COUNT),
      prev_resting_hr = if_else(is.na(prev_resting_hr), prev_resting_hr_, prev_resting_hr),
      prev_rmssd = if_else(is.na(prev_rmssd), prev_rmssd_, prev_rmssd),
      pre_intern_sleep_count = pre_intern_sleep_count^(1 / 2),
      pre_intern_step_count = pre_intern_step_count^(1 / 3)
    )

  common_control_vars <- paste(c(
    "Sex", "PHQtot0", "Neu0", "depr0", "EFE0", "pre_intern_step_count",
    "pre_intern_sleep_count", "pre_intern_resting_hr",
    "pre_intern_rmssd"
  ), collapse = "+")

  formula_backbone <- paste("~ msg_sent", common_control_vars, sep = "+")
  all_vars <- c("STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd")
  for (j in 1:length(all_vars)) {
    v <- all_vars[j]
    print(v)
    # f = paste(v,formula_backbone,"+",paste("prev",v,sep="_"),sep=" ")
    f <- paste(v, formula_backbone, sep = " ")
    f <- as.formula(f)
    for (k in 2:max(tmp$study_week)) {
      print(k)
      tmp_outcome <- tmp %>%
        filter(study_week == k) %>%
        select(USERID, all_of(v))
      for (l in 1:(k - 1)) {
        tmp_baseline <- tmp %>%
          select(USERID, Sex, PHQtot0, Neu0, depr0, EFE0, starts_with("pre_intern")) %>%
          unique()
        tmp_action <- tmp %>%
          filter(study_week == k - l) %>%
          select(USERID, msg_sent)
        tmp1 <- tmp_outcome %>%
          inner_join(tmp_action) %>%
          inner_join(tmp_baseline)
        fit <- lm(f, data = tmp1)
        res <- res %>% bind_rows(data.frame(est = fit$coefficients["msg_sent"], week = k, imp = i, mediator = v, lag = l))
      }
    }
  }
  res
})

results3 %>%
  group_by(mediator, week, imp) %>%
  summarize(est_sum = sum(est)) %>%
  arrange(mediator, week) %>%
  group_by(mediator, imp) %>%
  mutate(est_cummean = cummean(est_sum)) %>%
  group_by(mediator, week) %>%
  summarize(effect = mean(est_cummean)) %>%
  group_by(mediator) %>%
  group_modify(fun) %>%
  ggplot(aes(x = week, y = effect_smooth)) +
  theme_bw() +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  xlab("Week") +
  ylab("Delayed effect of push notifications") +
  facet_wrap(~mediator,
    scales = "free_y", ncol = 4, strip.position = "top",
    labeller = labeller(mediator = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  )


p17_delayed <- results3 %>%
  arrange(mediator, week) %>%
  group_by(mediator, imp, week) %>%
  summarise(est_week = sum(est)) %>%
  group_by(mediator, imp) %>%
  mutate(est_cummean = cummean(est_week)) %>%
  group_by(mediator, week) %>%
  summarize(effect = mean(est_cummean)) %>%
  group_by(mediator) %>%
  group_modify(fun) %>%
  ggplot(aes(x = week, y = effect_smooth)) +
  theme_bw() +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  xlab("Week") +
  ylab("Delayed effect of push notifications") +
  facet_wrap(~mediator,
    scales = "free_y", ncol = 4, strip.position = "top",
    labeller = labeller(mediator = c(
      "rmssd" = "HRV", "SLEEP_COUNT" = "Sleep",
      "STEP_COUNT" = "Step", "resting_hr" = "RHR",
      "MOOD" = "Mood"
    ))
  ) + 
  theme(axis.title.y = element_blank())
p17_delayed

###########################################################################################

# ggsave("~/Downloads/ihs2018_excursion_effect.eps",plot = p6,device = cairo_ps,fallback_resolution=300,width = 27,height = 9,units = "cm")
ggsave("~/Downloads/ihs2018_excursion_effect_mi_delayed.png", plot = p17_delayed, width = 23, height = 7, units = "cm")

ggsave("~/Downloads/ihs2018_excursion_effect_mi_immediate.png", plot = p16_immediate, width = 23, height = 7, units = "cm")

png("~/Downloads/ihs2018_activity_msg_effects.png", width = 800, height = 700, res = 200)
p11
dev.off()

png("~/Downloads/ihs2018_sleep_msg_effects.png", width = 800, height = 700, res = 200)
p22
dev.off()

png("~/Downloads/ihs2018_mood_msg_effects.png", width = 800, height = 700, res = 200)
p33
dev.off()



#######################################################################################################################
load("~/Dropbox (University of Michigan)/ihs_data/2018/imputed_data/imputation_list_daily_separated_no_mess_sep.RData")
dat <- impute_list[[9]]$full_data_complete %>%
  select(-c(NOTIFICATION_TYPE, date_day)) %>%
  mutate(week_category = if_else(week_category == "unsure", "None", week_category)) %>%
  group_by(USERID, study_week, week_category) %>%
  summarise(across(everything(), ~ mean(.x))) %>%
  ungroup() %>%
  left_join(impute_list[[9]]$all_baseline)
tmp <- dat %>%
  mutate(msg_sent = (week_category == "mood") + 0) %>%
  arrange(USERID, study_week) %>%
  group_by(USERID) %>%
  mutate(prev_MOOD = lag(MOOD, 1))
tmp <- tmp %>%
  select(USERID, pre_intern_mood) %>%
  unique() %>%
  rename(prev_mood = "pre_intern_mood") %>%
  mutate(study_week = 1) %>%
  right_join(tmp) %>%
  mutate(prev_MOOD = if_else(is.na(prev_MOOD), prev_mood, prev_MOOD))


formula_backbone <- "~ study_week * msg_sent + Neu0 + depr0 + EFE0 + pre_intern_mood + prev_MOOD"
all_mediators <- c("MOOD")
res <- tibble()
for (i in 1:length(all_mediators)) {
  mediator <- all_mediators[i]
  print(mediator)
  f <- paste(mediator, formula_backbone, sep = " ")
  f <- as.formula(f)
  fit <- geeglm(f, data = tmp, id = USERID, corstr = "i")


  result <- my_clusbootglm(f,
    data = tmp, family = gaussian(link = "identity"), clusterid = USERID, B = 1000, n.cores = 8
  )
  v <- result$coefficients
  res0 <- t(rbind(
    result$lm.coefs["msg_sent"] + 0:25 * result$lm.coefs["study_week:msg_sent"],
    apply(replicate(26, v[, "msg_sent"]) + sapply(0:25, function(i) i * v[, "study_week:msg_sent"]),
      2,
      FUN = function(x) quantile(x, c(0.025, 0.975))
    )
  )) %>% data.frame()
  colnames(res0) <- c("effect", "effect_lb", "effect_ub")
  res0 <- res0 %>%
    tibble() %>%
    mutate(week = 0:25, response = rep(mediator, 26))
  res <- res %>% bind_rows(res0)
}

ggplot(data = res, aes(x = week, y = effect)) +
  theme_bw() +
  geom_ribbon(aes(ymin = effect_lb, ymax = effect_ub), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_color_manual(values = c("effect" = "black", "effect_lb" = "blue", "effect_ub" = "blue")) +
  scale_linetype_manual(values = c("effect" = "solid", "effect_lb" = "dotted", "effect_ub" = "dotted")) +
  xlab("Week") +
  ylab("Effect of message on MOOD")
