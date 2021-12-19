    library(tidyverse)

    Sys.setenv("_R_USE_PIPEBIND_" = "true")
    in_file <- Sys.getenv("IN_FILE")
    out_file <- Sys.getenv("OUT_FILE")

    in_file <- "/Users/soren/Repositories/Speciale/notebooks/tmp.csv" 
    out_file <- "/Users/soren/Repositories/Speciale/thesis/Figures/mnist-calibration.pdf"

    data <- read_csv(in_file) |>
        mutate(plot_idx=case_when(
            interval == "(0.0, 0.01]" ~ 1,
            interval == "(0.99, 1.0]" ~ 3,
            TRUE ~ 2
        )) 

    make_plot <- function(data) {
        data |>
            mutate(sd=sqrt(proportion*(1-proportion)/count)) |>
            ggplot(aes(x=interval, y=proportion, col=inference)) +
            geom_errorbar(aes(ymin=lower,ymax=upper,group=1), col="black", linetype="dashed") +
            geom_point() + 
            geom_line(aes(group=inference)) +
            ggpubr::theme_pubr()
    }

    get_legend <- function(myggplot){
    tmp <- ggplot_gtable(ggplot_build(myggplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)
    }

    plots <- map(c(1,2, 3), ~ make_plot(filter(data, plot_idx==.x))) 
    legend <- get_legend(plots[[2]])
    plots <- map(plots, ~ .x + theme(legend.position="none"))
    gridExtra::arrangeGrob(
        grobs=list(
        plots[[1]], 
        plots[[2]] + theme(axis.text.x = element_text(angle = 45, hjust=1)), 
        plots[[3]]
        ), 
        layout_matrix=rbind(c(2, 3),c( 2, 1)), 
        widths=c(2,1)
        ) |>
        gridExtra::grid.arrange(legend, heights=c(10,1)) |> p =>
        ggsave(out_file, plot=p, height=6, width=8)

        



