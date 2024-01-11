library(generalCorr)

nlgc <- function(x1, x2, pwanted=4, px1=4, px2=4, n999=9) {
    x1 = as.numeric(x1)
    x2 = as.numeric(x2)
    # print(length(x1))
    # print(class(x1))
    # print(x1)
    # print(class(x2))
    # print(x2)
    bootGcRsq(x1, x2, px1=px1, px2=px2, pwanted=pwanted, n999=n999)
}