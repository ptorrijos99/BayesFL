/*
 *  The MIT License (MIT)
 *  
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */
/**
 *    LocalExperiment.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.experiments;

import bayesfl.algorithms.BN_GES;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.MultipleConvergence;
import bayesfl.convergence.ScoreImprovement;
import bayesfl.data.BN_DataSet;
import bayesfl.fusion.BN_FusionIntersection;
import bayesfl.fusion.BN_FusionUnion;
import bayesfl.fusion.Fusion;
import edu.cmu.tetrad.data.DataSet;
import bayesfl.Client;
import bayesfl.Server;
import bayesfl.convergence.ModelEquality;
import org.albacete.simd.utils.Utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;

public class LocalExperiment {
    public static String PATH = "./";
    
    public static void main(String[] args) {
        simpleExperiment();
        //multipleExperiment();
    }
    
    public static void simpleExperiment() {
        String net = "andes";
        String algName = "GES";
        String convergence = "Multiple";  // "Multiple" "Score" "Model"
        String fusionClient = "Union";
        String fusionServer = "MinCut";  //"Union" "Consensus" "GeneticTW" "MaxTreewidth" "MaxFrequency" "MinCut"
        
        String maxEdgesIt = "5";  // "10" "0.15" "log2" "log10" "sqrt2" "sqrt3"...

        int nIterations = 150;

        //int maxEdgesIt = 10000000;
        //int nIterations = 1;

        //String[] bbdd_paths = new String[]{"0", "1", "2", "3"};
        //launchExperiment(net, algName, convergence, fusionClient, fusionServer, bbdd_paths, maxEdgesIt, nIterations);
        
        String bbdd = "-1";
        int nClients = 10;

        String limitC = "-1";
        String limitS = "0.0";

        double alpha = 0;  // Controls the no-IID level of the data distribution

        launchExperiment(net, algName, convergence, fusionClient, limitC, fusionServer, limitS, bbdd, nClients, maxEdgesIt, nIterations, alpha);
    }


    public static void launchExperiment(String net, String algName, String convergence, String fusionC, String limitC, String fusionS, String limitS, String bbdd, int nClients, String maxEdgesIt, int nIterations, double alpha) {
        DataSet allData = null;
        ArrayList<DataSet> divisionData = null;
        int maxEdgesItInt;

        String ampliedPath;
        if (alpha == 0.0) ampliedPath = PATH + "res/networks/BBDD/" + net + "/" + net + ".";
        else ampliedPath = PATH + "res/networks/BBDD_noIID/" + net + "/" + alpha + "/" + net + ".";

        if (!Objects.equals(bbdd, "-1")) {
            allData = Utils.readData(ampliedPath + bbdd + ".csv");
            maxEdgesItInt = matchMaxEdgesIt(maxEdgesIt, allData.getNumColumns());
            System.out.println("Nodes: " + allData.getNumColumns() + ", Limit: " + maxEdgesIt + ", Max Edges: " + maxEdgesItInt + "\n");
        } else {
            divisionData = new ArrayList<>();
            for (int i = 0; i < nClients; i++) {
                DataSet data = Utils.readData(ampliedPath + i + ".csv");
                divisionData.add(data);
            }
            maxEdgesItInt = matchMaxEdgesIt(maxEdgesIt, divisionData.get(0).getNumColumns());
            System.out.println("Nodes: " + divisionData.get(0).getNumColumns() + ", Limit: " + maxEdgesIt + ", Max Edges: " + maxEdgesItInt + "\n");
        }

        String operation = algName + "," + maxEdgesIt+"="+maxEdgesItInt + "," + fusionC + "," + limitC + "," + convergence + "," + fusionS + "," + limitS + "," + alpha;
        String savePath = "./results/Server/" + net + "." + bbdd + "_" + operation + "_" + nClients + "_-1.csv";

        if (!checkExistentFile(savePath)) {
            if (!Objects.equals(bbdd, "-1")) divisionData = BN_DataSet.divideDataSet(allData, nClients);

            ArrayList<Client> clients = new ArrayList<>();

            for (int i = 0; i < nClients; i++) {
                BN_DataSet data = new BN_DataSet(divisionData.get(i), (net + "." + bbdd + "." + i));

                Fusion fusionClient;
                if (fusionC.equals("BN_FusionIntersection")) {
                    fusionClient = new BN_FusionIntersection();
                } else {
                    fusionClient = new BN_FusionUnion();
                    ((BN_FusionUnion) fusionClient).setMode(fusionC);
                    ((BN_FusionUnion) fusionClient).setLimit(limitC);
                }

                data.setOriginalBNPath(PATH + "res/networks/" + net + ".xbif");
                LocalAlgorithm algorithm = new BN_GES(algName, "None", maxEdgesItInt);

                Client client = new Client(fusionClient, algorithm, data);
                client.setStats(true, true, PATH);
                client.setExperimentName(operation);
                clients.add(client);
            }

            Fusion fusionServer;
                if (fusionS.equals("BN_FusionIntersection")) {
                    fusionServer = new BN_FusionIntersection();
                } else {
                    fusionServer = new BN_FusionUnion();
                    ((BN_FusionUnion) fusionServer).setMode(fusionS);
                    ((BN_FusionUnion) fusionServer).setLimit(limitS);
                }

            Convergence conv;
            if (convergence.equals("Score")) {
                conv = new ScoreImprovement();
            } else if (convergence.equals("Model")) {
                conv = new ModelEquality();
            }
            else {
                Convergence[] convs = new Convergence[2];
                convs[0] = new ScoreImprovement();
                convs[1] = new ModelEquality();
                conv = new MultipleConvergence(convs);
            }
            Server server = new Server(fusionServer, conv, clients);

            server.setStats(true, PATH);
            BN_DataSet data = new BN_DataSet(net + "." + bbdd);
            data.setOriginalBNPath(PATH + "res/networks/" + net + ".xbif");
            server.setData(data);

            server.setExperimentName(operation);
            server.setnIterations(nIterations);

            server.run();
            
            writeExistentFile(savePath);
        } else {
            System.out.println("\n EXISTENT EXPERIMENT: " + savePath + "\n");
        }
    }

    public static int matchMaxEdgesIt(String maxEdgesIt, int nNodes) {
        int maxEdgesItInt;
        if (maxEdgesIt.matches("\\d+")) {
            // It's a positive integer
            maxEdgesItInt = Integer.parseInt(maxEdgesIt);
        } else if (maxEdgesIt.matches("\\d*\\.\\d+")) {
            // It's a decimal
            double maxEdgesItDouble = Double.parseDouble(maxEdgesIt);
            if (maxEdgesItDouble > 1 || maxEdgesItDouble <= 0) maxEdgesItDouble = 1;
            maxEdgesItInt = (int) Math.round(nNodes * maxEdgesItDouble);
        } else if (maxEdgesIt.startsWith("log")) {
            // Logarithm in base x of nNodes
            double base = 2;
            if (maxEdgesIt.length() > 3) { // Check if there is a number after "log"
                base = Double.parseDouble(maxEdgesIt.substring(3).trim());
            }
            if (base <= 0 || base == 1) throw new IllegalArgumentException("log(x): x must be > 0 y != 1");
            maxEdgesItInt = (int) Math.round(Math.log(nNodes) / Math.log(base)); // Cambio de base
        } else if (maxEdgesIt.startsWith("sqrt")) {
            // Root in base x of nNodes
            double base = 2;
            if (maxEdgesIt.length() > 4) { // Check if there is a number after "sqrt"
                base = Double.parseDouble(maxEdgesIt.substring(4).trim());
            }
            if (base <= 0) throw new IllegalArgumentException("sqrt(x): x must be > 0");
            maxEdgesItInt = (int) Math.round(Math.pow(nNodes, 1.0 / base)); // Root in base x
        } else {
            throw new IllegalArgumentException("Invalid format for maxEdgesIt");
        }
        return maxEdgesItInt;
    }

    public static boolean checkExistentFile(String savePath) {
        File file = new File(savePath);
        
        return file.length() != 0;
    }
    
    public static void writeExistentFile(String savePath) {
        File file = new File(savePath);
        file.getParentFile().mkdirs();

        BufferedWriter csvWriter = null;
        try {
            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            csvWriter.append(" ");
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }
    }
}
