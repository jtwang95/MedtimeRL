library(tidyverse)
library(ggplot2)

d <- read_csv("../outs/nonstationarity_all_all/param_est_MI_finite_all.csv")

dat <- d %>% filter(idx == 0 & j == 0)

dat %>% ggplot(aes(x=t,y=value,group=t)) +
    geom_boxplot(outlier.shape = NA) + 
    facet_wrap(~key,ncol=3,scales="free") +
    # theme_minimal() +
    xlab("Week") +
    ylab("Estimate") -> p1
ggsave("../outs/nonstationarity_all_all/param_est_MI_finite_all.png",plot=p1,units="cm",width=18,height=12)
    
