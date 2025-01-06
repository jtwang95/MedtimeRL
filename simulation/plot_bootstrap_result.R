library(tidyverse)
library(ggplot2)
library(ggpubr)

d0 <- read_csv("./outs/finite_bs_exp6_d3_seed1_240227235021.csv") # finite
d1 <- read_csv("./outs/infinite_bs_exp6_d3_seed1_240304160829.csv") # infinite

################################################################################
# Finite
dat0 <- d0 %>%
    group_by(label,j) %>%
    summarise(est_mean = mean(est),
              bias_mean = mean(est-true),
              cp = mean(covered),
              mean_bstd = mean(bstd),
              ese = sd(est))

p0 <- ggplot(dat0, aes(x=j, y=cp, color = label)) +
  geom_point() +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_hline(yintercept = 0.95, linetype="dashed", color = "black") +
  labs(title="Finite", x="Mediator", y="Converage probability")
ggsave(filename="./figures/finite_bootstrap_cp.png",plot=p0,units="cm",width=10,height=9)

################################################################################# 
# Infinite
dat1 <- d1 %>%
    group_by(label,j) %>%
    summarise(est_mean = mean(est),
              bias_mean = mean(est-true),
              cp = mean(covered),
              mean_bstd = mean(bstd),
              ese = sd(est))

p1 <- ggplot(dat1, aes(x=j, y=cp, color = label)) +
  geom_point() +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_hline(yintercept = 0.95, linetype="dashed", color = "black") +
  labs(title="Finite", x="Mediator", y="Converage probability")
ggsave(filename="./figures/ininite_bootstrap_cp.png",plot=p1,units="cm",width=10,height=9)

#################################################################################
# Combine plots
dat <- dat0 %>% mutate(setting = "Finite horizon setting") %>% bind_rows(dat1 %>% mutate(setting = "Infinite horizon setting")) %>%
    mutate(Method = case_when(label == "ours" ~ "Proposed method",
                             label == "time_independence" ~ "Independent time points",
                             label == "conditional_independence" ~ "Independent mediators")) %>%
    mutate(Method = factor(Method, levels = c("Proposed method", "Independent time points", "Independent mediators")))
p2 <- dat %>% ggplot(aes(x=j, y=cp, color = Method, shape = Method)) +
  geom_hline(yintercept = 0.95, linetype="dashed", color = "black") +
  geom_point(position = position_dodge(width = 0.5)) +
#   theme_minimal() +
  theme(legend.position = "bottom") +
  scale_x_continuous(breaks = c(0,1,2)) +
  scale_y_continuous(breaks = c(0.95,seq(0,1,0.25))) +
  labs(x="Mediator index", y="Converage probability") + 
  facet_wrap(~setting, nrow = 1) +
  theme(legend.position = "right")
ggsave(filename="./figures/combined_bootstrap_cp.png",plot=p2,units="cm",width=15,height=7)
