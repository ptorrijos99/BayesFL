package bayesfl.experiments.utils;

import java.io.File;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class SaveBBDDsMetadata {

    public static void main(String[] args) throws Exception {

        // Directorio de los archivos .arff
        File directory = new File("./res/classification/");
        File[] subfolders = directory.listFiles(File::isDirectory);

        // Crear archivo CSV para escribir los metadatos
        FileWriter csvWriter = new FileWriter("./res/classification/metadata-.csv");
        csvWriter.append("name,subfolder,nInstances,nAttributes,nClasses,discAtts,numAtts\n");

        // Leer los archivos .arff en cada subcarpeta y calcular los metadatos
        for (File subfolder : subfolders) {
            File[] arffFiles = subfolder.listFiles((dir, name) -> name.endsWith(".arff"));
            for (File arffFile : arffFiles) {
                System.out.println("Processing file: " + subfolder.getName() + "/" + arffFile.getName());

                String name = arffFile.getName();
                String subfolderName = subfolder.getName();

                ArffLoader loader = new ArffLoader();
                loader.setFile(arffFile);
                Instances data = loader.getDataSet();
                data.setClassIndex(data.numAttributes()-1);

                int nInstances = data.numInstances();
                int nAttributes = data.numAttributes();
                int nClasses = data.numClasses();

                int discAtts = 0;
                int numAtts = 0;
                for (int i = 0; i < nAttributes; i++) {
                    if (data.attribute(i).isNumeric()) {
                        numAtts++;
                    } else {
                        discAtts++;
                    }
                }

                String metadata = name + "," + subfolderName + "," + nInstances + "," +
                        nAttributes + "," + nClasses + "," + discAtts + "," + numAtts + "\n";
                System.out.println(metadata);

                // Escribir los metadatos en el archivo CSV
                csvWriter.append(metadata);

                csvWriter.flush();
            }
        }

        // Cerrar el archivo CSV
        csvWriter.flush();
        csvWriter.close();
    }

}
