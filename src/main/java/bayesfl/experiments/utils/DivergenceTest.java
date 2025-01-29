package bayesfl.experiments.utils;

import bayesfl.data.BN_DataSet;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import org.albacete.simd.utils.Utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class DivergenceTest {

    public static String PATH = "./";

    public static void main(String[] args) {
        String folderPath = "res/networks/BBDD_old";
        File folder = new File(folderPath);

        // Get all .bif files and remove the extension
        List<String> nets = new ArrayList<>();
        File[] files = folder.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile() && file.getName().endsWith(".ALL.csv")) {
                    String netName = file.getName().replace(".ALL.csv", "");
                    nets.add(netName);
                }
            }
        }

        File newFolder = new File(PATH + "results/divergence/");
        if (!newFolder.exists()) {
            newFolder.mkdirs();
        }

        //List<String> nets = Arrays.asList("earthquake");

        for (String net : nets) {
            String path = PATH + "results/divergence/" + net + ".csv";
            // Check that the file does not exist
            File file = new File(path);
            if (file.exists()) {
                System.out.println("The file already exists");
                continue;
            }

            divergence(net);
        }
    }

    public static void divergence(String net) {
        List<Integer> nClients = Arrays.asList(1, 5, 10, 20, 30, 50, 100);

        ArrayList<String> distancesToCompute = new ArrayList<>();
        distancesToCompute.add("EMD");
        distancesToCompute.add("EMD2");
        distancesToCompute.add("Mallows");
        distancesToCompute.add("Mallows2");
        distancesToCompute.add("Bhattacharyya");
        distancesToCompute.add("KL");
        distancesToCompute.add("JS");



        System.out.println("\nReading all data of " + net);
        DataSet allData = Utils.readData(PATH + "res/networks/BBDD_old/" + net + ".ALL.csv");

        // Get the global frequencies
        double[][] matrixData = allData.getDoubleData().toArray();
        Map<String, Map<String, Integer>> globalFrequencies = getUniqueValuesWithCounts(matrixData, allData);

        double[][] distances = new double[nClients.size()+1][distancesToCompute.size()];

        // Iterate over each seed bbdd
        for (int seed = 0; seed <= 9; seed++) {
            System.out.println("Reading seed " + seed + " of " + net);
            DataSet dataSeed = Utils.readData(PATH + "res/networks/BBDD_old/" + net + "." + seed + ".csv");

            // Get the frequencies for the current seed dataset
            double[][] clientMatrixData = dataSeed.getDoubleData().toArray();
            Map<String, Map<String, Integer>> clientFrequencies = getUniqueValuesWithCounts(clientMatrixData, allData);

            // Use the metrics
            for (String distance : distancesToCompute) {
                double metric = computeMetric(distance, globalFrequencies, clientFrequencies);
                distances[0][distancesToCompute.indexOf(distance)] += metric;
            }

            // Iterate over different client divisions
            for (int i = 1; i < nClients.size(); i++) {
                int nClient = nClients.get(i);
                ArrayList<DataSet> divisionData = BN_DataSet.divideDataSet(dataSeed, nClient);

                // For each divided dataset
                for (int j = 0; j < nClient; j++) {
                    DataSet clientData = divisionData.get(j);

                    // Get the frequencies for the current client dataset
                    clientMatrixData = clientData.getDoubleData().toArray();
                    clientFrequencies = getUniqueValuesWithCounts(clientMatrixData, allData);

                    // Use the metrics
                    for (String distance : distancesToCompute) {
                        double metric = computeMetric(distance, globalFrequencies, clientFrequencies);
                        distances[i][distancesToCompute.indexOf(distance)] += metric;
                    }
                }
            }
        }

        for (int i = 0; i < nClients.size(); i++) {
            for (int j = 0; j < distancesToCompute.size(); j++) {
                distances[i][j] /= (10 * nClients.get(i));
            }

            System.out.println("Clients: " + nClients.get(i) + " - EMD: " + distances[i][0] + " - EMD2: " + distances[i][1] + " - Mallows: " + distances[i][2] + " - Mallows2: " + distances[i][3] + " - Bhattacharyya: " + distances[i][4] + " - KL: " + distances[i][5] + " - JS: " + distances[i][6]);
        }

        saveToCSV(net, nClients, distances, distancesToCompute);
    }

    public static void saveToCSV(String net, List<Integer> nClients, double[][] distances, ArrayList<String> distancesToCompute) {
        String path = PATH + "results/divergence/" + net + ".csv";

        StringBuilder sb = new StringBuilder();
        sb.append("Net,Clients,");
        for (int i = 0; i < distances[0].length; i++) {
            sb.append(distancesToCompute.get(i));
            if (i < distances[0].length - 1) {
                sb.append(",");
            }
        }
        sb.append("\n");

        for (int i = 0; i < nClients.size(); i++) {
            sb.append(net).append(",").append(nClients.get(i)).append(",");
            for (int j = 0; j < distances[0].length; j++) {
                sb.append(distances[i][j]);
                if (j < distances[0].length - 1) {
                    sb.append(",");
                }
            }
            sb.append("\n");
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            writer.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static Map<String, Map<String, Integer>> getUniqueValuesWithCounts(double[][] matrix, DataSet allData) {
        Map<String, Map<String, Integer>> result = obtainEmptyMap(allData);
        List<String> columnNames = allData.getVariableNames();

        // Iterate over each column in the matrix
        for (int col = 0; col < matrix[0].length; col++) {
            String columnName = columnNames.get(col); // Get the current column name
            Map<String, Integer> counts = result.get(columnName); // Get the count map for the column

            // Get the DiscreteVariable corresponding to the column
            DiscreteVariable variable = (DiscreteVariable) allData.getVariable(columnName);

            // Iterate through each row and update the count for the current column value
            for (double[] doubles : matrix) {
                // Get the index of the category corresponding to the double value in the matrix
                int categoryIndex = (int) doubles[col];

                String category = variable.getCategories().get(categoryIndex); // Get the category name
                counts.put(category, counts.getOrDefault(category, 0) + 1); // Increment the count for the category
            }
        }

        return result; // Return the final result
    }

    public static Map<String, Map<String, Integer>> obtainEmptyMap(DataSet allData) {
        Map<String, Map<String, Integer>> sampleHashMap = new HashMap<>();
        for (String attribute : allData.getVariableNames()) {
            sampleHashMap.put(attribute, new HashMap<>());
            DiscreteVariable var = (DiscreteVariable) allData.getVariable(attribute);
            for (String category : var.getCategories()) {
                sampleHashMap.get(attribute).put(category, 0);
            }
        }
        return sampleHashMap;
    }

    public static double computeMetric(String type, Map<String, Map<String, Integer>> globalFrequencies, Map<String, Map<String, Integer>> clientFrequencies) {
        HashMap<String, Double> result = new HashMap<>();

        // Iterate over each attribute (column)
        for (String attribute : globalFrequencies.keySet()) {
            double metric = 0.0;  // Initialize the metric value
            Map<String, Integer> globalCounts = globalFrequencies.get(attribute);
            Map<String, Integer> clientCounts = clientFrequencies.get(attribute);

            int totalGlobal = globalCounts.values().stream().mapToInt(Integer::intValue).sum();
            int totalClient = clientCounts.values().stream().mapToInt(Integer::intValue).sum();

            // Iterate over each category
            for (String category : globalCounts.keySet()) {
                double p = globalCounts.get(category) / (double) totalGlobal;  // Global probability
                double q = clientCounts.getOrDefault(category, 0) / (double) totalClient;  // Client probability
                double m;

                switch (type) {
                    case "EMD" -> metric += Math.abs(p - q);  // Earth Mover's distance formula
                    case "EMD2" -> metric += Math.abs(p - q) / 2;  // Earth Mover's distance formula / 2
                    case "Mallows" -> metric += Math.pow(p - q, 2);  // Mallows distance formula
                    case "Mallows2" -> metric += Math.pow(p - q, 2) / 2;  // Mallows distance formula / 2
                    case "Bhattacharyya" -> metric += Math.sqrt(p * q);  // Bhattacharyya distance formula
                    case "KL" -> {
                        if (p > 0 && q > 0) { // Avoid log(0) errors
                            metric += p * Math.log(p / q);  // KL divergence formula
                        }
                        else if (p > 0) {
                            metric += p * Math.log(p / 0.000000001);  // KL divergence formula with a small value to avoid log(0)
                            //metric += Double.POSITIVE_INFINITY;
                        }
                    }
                    case "JS" -> {
                        if (p > 0 && q > 0) { // Avoid log(0) errors
                            m = (p + q) / 2;  // Mixture probability
                            metric += (p * Math.log(p / m) + q * Math.log(q / m)) / 2;  // Jensen-Shannon divergence formula
                        }
                    }
                }
            }

            if (type.equals("Mallows") || type.equals("Mallows2")) {
                metric = Math.sqrt(metric);  // Compute the square root of the Mallows distance
            } else if (type.equals("Bhattacharyya")) {
                metric = -Math.log(metric);  // Compute the negative logarithm of the Bhattacharyya distance
            }

            result.put(attribute, metric);  // Store the metric value for the current attribute
        }

        double mean = result.values().stream().mapToDouble(Double::doubleValue).sum() / result.size();  // Compute the mean of all the variables
        return mean;
    }


}
