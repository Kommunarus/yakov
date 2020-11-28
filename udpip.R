#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
library(udpipe)
udmodel <- udpipe_download_model(language = "russian-syntagrus", overwrite=FALSE)
ss <- args[1]
my_data<-readChar(ss, file.info(ss)$size)
aa<-udpipe(x = my_data,  object = udmodel)
write.csv(aa,file='/home/alex/PycharmProjects/coref/outputR.csv')