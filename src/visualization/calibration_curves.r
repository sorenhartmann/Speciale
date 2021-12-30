    library(tidyverse)

    Sys.setenv("_R_USE_PIPEBIND_" = "true")
    in_file <- Sys.getenv("IN_FILE")
    out_file <- Sys.getenv("OUT_FILE")

    data <- read_csv(in_file) 

    plot <- data |>
        filter(count>5) |>
        ggplot(aes(x=mean_confidence, y=mean_accuracy, col=inference)) +
        geom_point() + 
        geom_abline(intercept=0, slope=1, linetype="dashed") +
        lims(x=c(0, 1),y=c(0,1))+
        labs(x="Confidence", y="Accuracy") +
        ggpubr::theme_pubr()

    plot <- ggpubr::ggarrange(
        plot+geom_line(), 
        plot+xlim(0.96, 1)+ylim(0.96, 1), 
        common.legend=TRUE, 
        ncol=2, 
        nrow=1
        ) 

    

    ggsave(out_file, plot=plot, height=4, width=8)

        



