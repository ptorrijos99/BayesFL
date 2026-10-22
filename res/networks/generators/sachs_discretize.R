# Discretises the flow-cytometry data of Sachs et al. (2005) into three levels
# (Hartemink), as used for the sachs network. Reads sachs.data.txt (the original
# continuous data) from the working directory and writes sachs.csv.

# Cargar bnlearn
library(bnlearn)

sachs <- read.table("sachs.data.txt", header = TRUE)
print(head(sachs))

# Discretizar los datos
dsachs <- discretize(sachs, method = "hartemink", breaks = 3, ibreaks = 60, idisc = "quantile")
print(head(dsachs))

# Verifica los niveles originales de Akt
print(levels(dsachs$Akt))


# Cambiar niveles a LOW, AVG, HIGH para todas las variables discretizadas
for (col in colnames(dsachs)) {
    # Asegurarse de que cada columna sea un factor
    dsachs[[col]] <- as.factor(dsachs[[col]])
    
    # Verificar los niveles actuales antes de cambiar
    print(paste("Niveles antes de cambiar en", col, ":", paste(levels(dsachs[[col]]), collapse = ", ")))
    
    # Cambiar los niveles
    levels(dsachs[[col]]) <- c("LOW", "AVG", "HIGH")
    
    # Imprimir niveles después de cambiar
    print(paste("Niveles después de cambiar en", col, ":", paste(levels(dsachs[[col]]), collapse = ", ")))
}

# Guardar el archivo discretizado como csv
write.csv(dsachs, "sachs.csv", row.names = FALSE)