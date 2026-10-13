#!/usr/bin/env Rscript
# Main-text Figure 1: bromodomain tree with PDB-structure bubbles.

suppressMessages({
  library(ggplot2)
  library(ggtree)
  library(ape)
  library(tidytree)
})

tree_dir <- "tree_inputs"
mydata <- read.table(file.path(tree_dir, "data.dat"), header = TRUE, sep = "\t")
tree <- read.tree(file.path(tree_dir, "1tree.tree"))

benchmark <- c("BRD4—1", "TRIM24", "SMARCA2", "CREBBP", "TAF1L—2",
               "ATAD2", "BRD9", "BAZ2A", "BRD2—2", "BPTF")
mydata$selected <- ifelse(mydata$label %in% benchmark, "yes", "no")
stopifnot(all(benchmark %in% tree$tip.label))

y <- full_join(as_tibble(tree), mydata, by = "label")

p <- ggtree(as.treedata(y), layout = "circular", branch.length = "none", size = 6) +
  geom_hilight(node = 98, fill = "#1B9E77") +
  geom_hilight(node = 101, fill = "#D95F02") +
  geom_hilight(node = 78, fill = "#7570B3") +
  geom_hilight(node = 81, fill = "#E7298A") +
  geom_hilight(node = 74, fill = "#66A61E") +
  geom_hilight(node = 108, fill = "#E6AB02") +
  geom_hilight(node = 96, fill = "#A6761D") +
  geom_hilight(node = 92, fill = "#666666") +
  geom_hilight(node = 68, fill = "blue") +
  geom_point(aes(size = PDB, fill = PDB), shape = 21) +
  scale_fill_gradient(low = "white", high = "red", guide = "legend") +
  scale_size_continuous(range = c(0, 30)) +
  geom_tiplab(aes(color = selected), fontface = "bold", offset = 0.7, size = 9) +
  scale_color_manual(values = c(no = "black", yes = "#d62728"), guide = "none", na.value = "black") +
  theme(legend.key.size = unit(1, "cm"),
        legend.title = element_text(size = 30, face = "bold"),
        legend.text = element_text(size = 20, face = "bold"))

ggsave("../figures/fig1_phylogeny.pdf", p, width = 20, height = 20, device = cairo_pdf)
cat("wrote ../figures/fig1_phylogeny.pdf\n")
