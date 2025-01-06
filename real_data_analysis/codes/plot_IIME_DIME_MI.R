library(mgcv)
library(splines)
library(tidyverse)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

dat <- read_csv(file.path(args[1],"/res_finite_all.csv"))

tmp = dat %>% select(t,IIME,DIME,IIME_q0,IIME_q1,DIME_q0,DIME_q1,mediator,imp_iter) %>%
  group_by(t,mediator) %>%
  summarise(IIME = mean(IIME),IIME_lb = mean(IIME_q0), IIME_ub = mean(IIME_q1),
            DIME = mean(DIME),DIME_lb = mean(DIME_q0), DIME_ub = mean(DIME_q1)) %>%
  ungroup()
ns_df = 5

fun = function(.x,.y){
  # IIME, IIME_LB, IIME_UB
  fit = gam(IIME~ns(t,df=ns_df),data=.x)
  # IIME_smooth = predict(fit,newdata = data.frame(t=.x$t))
  IIME_smooth = predict(fit)
  fit = gam(IIME_lb~ns(t,df=ns_df),data=.x)
  # IIME_lb_smooth = predict(fit,newdata = data.frame(t=.x$t))
  IIME_lb_smooth = predict(fit)
  fit = gam(IIME_ub~ns(t,df=ns_df),data=.x)
  # IIME_ub_smooth = predict(fit,newdata = data.frame(t=.x$t))
  IIME_ub_smooth = predict(fit)

  # DIME, DIME_LB, DIME_UB
  fit = gam(DIME~ns(t,df=ns_df),data=.x)
  # DIME_smooth = predict(fit,newdata = data.frame(t=.x$t))
  DIME_smooth = predict(fit)
  .x %>% mutate(IIME_smooth = IIME_smooth,DIME_smooth=DIME_smooth)
  fit = gam(DIME_lb~ns(t,df=ns_df),data=.x)
  # DIME_lb_smooth = predict(fit,newdata = data.frame(t=.x$t))
  DIME_lb_smooth = predict(fit)
  .x %>% mutate(IIME_smooth = IIME_smooth,DIME_smooth=DIME_smooth)
  fit = gam(DIME_ub~ns(t,df=ns_df),data=.x)
  # DIME_ub_smooth = predict(fit,newdata = data.frame(t=.x$t))
  DIME_ub_smooth = predict(fit)
  .x %>% mutate(IIME_smooth = IIME_smooth,
                IIME_lb_smooth = IIME_lb_smooth,
                IIME_ub_smooth = IIME_ub_smooth,
                DIME_smooth=DIME_smooth,
                DIME_lb_smooth = DIME_lb_smooth,
                DIME_ub_smooth = DIME_ub_smooth)
}

tmp = tmp %>% group_by(mediator) %>% group_modify(fun)

# p1 = tmp %>% select(t,mediator,IIME_smooth,DIME_smooth) %>% 
#   pivot_longer(cols = ends_with("smooth")) %>% mutate(name=factor(name,levels=c("IIME_smooth","DIME_smooth"))) %>%
#   ggplot(aes(x=t,y=value,color=mediator)) + geom_line() +
#   theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") + 
#   facet_wrap(~name,ncol = 2)
ylim = c(-0.15,0.10)
# ylim = c(-0.015,0.01)

# IIME plot
p1 = tmp %>% select(t,mediator,starts_with("IIME")) %>% 
  mutate(mediator = case_match(mediator,"STEP_COUNT" ~ "Step","SLEEP_COUNT" ~ "Sleep","resting_hr" ~ "RHR", "rmssd"~"HRV")) %>%
  ggplot(aes(x=t,y=IIME_smooth,color=mediator,group=mediator)) + geom_line() + 
  geom_ribbon(aes(ymin=IIME_lb_smooth,ymax=IIME_ub_smooth),alpha=0.2,linetype="dashed") +
  theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") +
  ylim(ylim) +
  labs(x="Week",y="IIME") + 
  facet_wrap(~mediator,ncol = 2) + theme(legend.position = "none")

# DIME plot
# tmp = tmp %>% mutate(DIME_lb_smooth = if_else(t == 0,0,DIME_lb_smooth),DIME_ub_smooth = if_else(t == 0,0,DIME_ub_smooth))
p2 = tmp %>% filter(t>=1) %>% select(t,mediator,starts_with("DIME")) %>% 
  mutate(mediator = case_match(mediator,"STEP_COUNT" ~ "Step","SLEEP_COUNT" ~ "Sleep","resting_hr" ~ "RHR", "rmssd"~"HRV")) %>%
  ggplot(aes(x=t,y=DIME_smooth,color=mediator,group=mediator)) + geom_line() + 
  geom_ribbon(aes(ymin=DIME_lb_smooth,ymax=DIME_ub_smooth),alpha=0.2,linetype="dashed") +
  theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") +
  ylim(ylim) +
  labs(x="Week",y="DIME") + 
  facet_wrap(~mediator,ncol = 2) + theme(legend.position = "none")

# p2 = tmp %>% select(t,mediator,DIME_smooth) %>% filter(t>=1) %>%
#   rename(Week="t",DIME = "DIME_smooth") %>% 
#   mutate(mediator = case_match(mediator,"STEP_COUNT" ~ "Step","SLEEP_COUNT" ~ "Sleep","resting_hr" ~ "RHR", "rmssd"~"HRV")) %>%
#   ggplot(aes(x=Week,y=DIME,color=mediator)) + geom_line() +
#   theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") +
#   theme(legend.position = "right") + 
#   ylim(ylim)
# ggsave(filename=file.path(args[1],"ihs2018_IIME_smooth_r.png"),plot=p1,units="cm",width=10,height=9)
# ggsave(filename=file.path(args[1],"ihs2018_DIME_smooth_r.png"),plot=p2,units="cm",width=10,height=9)
ggsave(filename=file.path(args[1],"ihs2018_IIME_smooth_r.png"),plot=p1,units="cm",width=18,height=12,scale=0.8)
ggsave(filename=file.path(args[1],"ihs2018_DIME_smooth_r.png"),plot=p2,units="cm",width=18,height=12,scale =0.8)

ylim = c(-0.3,0.3)
p3 = tmp %>% select(t,mediator,starts_with("IIME")) %>% 
  mutate(mediator = case_match(mediator,"STEP_COUNT" ~ "Step","SLEEP_COUNT" ~ "Sleep","resting_hr" ~ "RHR", "rmssd"~"HRV")) %>%
  ggplot(aes(x=t,y=IIME,color=mediator,group=mediator)) + geom_line() + 
  geom_ribbon(aes(ymin=IIME_lb,ymax=IIME_ub),alpha=0.2,linetype="dashed") +
  theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") +
  ylim(ylim) +
  labs(x="Week",y="IIME") + 
  facet_wrap(~mediator,ncol = 2) + theme(legend.position = "none")

# DIME plot
# tmp = tmp %>% mutate(DIME_lb_smooth = if_else(t == 0,0,DIME_lb_smooth),DIME_ub_smooth = if_else(t == 0,0,DIME_ub_smooth))
p4 = tmp %>% filter(t>=1) %>% select(t,mediator,starts_with("DIME")) %>% 
  mutate(mediator = case_match(mediator,"STEP_COUNT" ~ "Step","SLEEP_COUNT" ~ "Sleep","resting_hr" ~ "RHR", "rmssd"~"HRV")) %>%
  ggplot(aes(x=t,y=DIME,color=mediator,group=mediator)) + geom_line() + 
  geom_ribbon(aes(ymin=DIME_lb,ymax=DIME_ub),alpha=0.2,linetype="dashed") +
  theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") +
  ylim(ylim) +
  labs(x="Week",y="DIME") + 
  facet_wrap(~mediator,ncol = 2) + theme(legend.position = "none")

# p2 = tmp %>% select(t,mediator,DIME_smooth) %>% filter(t>=1) %>%
#   rename(Week="t",DIME = "DIME_smooth") %>% 
#   mutate(mediator = case_match(mediator,"STEP_COUNT" ~ "Step","SLEEP_COUNT" ~ "Sleep","resting_hr" ~ "RHR", "rmssd"~"HRV")) %>%
#   ggplot(aes(x=Week,y=DIME,color=mediator)) + geom_line() +
#   theme_bw() + geom_hline(yintercept = 0.0,color="black",linetype="dashed") +
#   theme(legend.position = "right") + 
#   ylim(ylim)
# ggsave(filename=file.path(args[1],"ihs2018_IIME_smooth_r.png"),plot=p1,units="cm",width=10,height=9)
# ggsave(filename=file.path(args[1],"ihs2018_DIME_smooth_r.png"),plot=p2,units="cm",width=10,height=9)
ggsave(filename=file.path(args[1],"ihs2018_IIME_nonsmooth_r.png"),plot=p3,units="cm",width=18,height=12,scale=0.8)
ggsave(filename=file.path(args[1],"ihs2018_DIME_nonsmooth_r.png"),plot=p4,units="cm",width=18,height=12,scale =0.8)
