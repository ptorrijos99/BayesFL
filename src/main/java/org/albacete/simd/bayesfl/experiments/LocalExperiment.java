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
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.bayesfl.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import edu.cmu.tetrad.data.DataSet;

import org.albacete.simd.bayesfl.Client;
import org.albacete.simd.bayesfl.Server;
import org.albacete.simd.bayesfl.algorithms.BN_GES;
import org.albacete.simd.bayesfl.algorithms.LocalAlgorithm;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.fusion.BN_FusionUnion;
import org.albacete.simd.bayesfl.fusion.BN_FusionIntersection;
import org.albacete.simd.bayesfl.fusion.Fusion;

import static org.albacete.simd.bayesfl.data.BN_DataSet.divideDataSet;
import static org.albacete.simd.bayesfl.data.BN_DataSet.readData;

public class LocalExperiment {
    public static void main(String[] args) {
        simpleExperiment();
        //multipleExperiment();
    }
    
    public static void simpleExperiment() {
        String net = "andes";
        String algName = "GES";
        String refinement = "None";
        String fusionClient = "BN_FusionUnion";
        String fusionServer = "BN_FusionUnion";
        
        int maxEdgesIt = Integer.MAX_VALUE;
        int nIterations = 100;

        //String[] bbdd_paths = new String[]{"0", "1", "2", "3"};
        //launchExperiment(net, algName, refinement, fusionClient,fusionServer, bbdd_paths, maxEdgesIt, nIterations);
        
        String bbdd = "0";
        int nClients = 20;
        launchExperiment(net, algName, refinement, fusionClient,fusionServer, bbdd, nClients, maxEdgesIt, nIterations);
    }
    
    public static void multipleExperiment() {
        String[] nets = new String[]{"child", "water", "insurance", "alarm", "hailfinder", "hepar2", "mildew", "barley", "win95pts", "pathfinder", "andes", "pigs", "diabetes", "link", "munin"};
        
        //String[] bbdd_paths = new String[]{"0", "1", "2", "3"};
        String bbdd = "0";
        int[] nClients = {2, 4, 6, 8, 10, 15, 20};
        String[] algNames = new String[]{"pGES", "GES"};
        String[] refinements = new String[]{"None"};
        int[] maxEdgesIts = new int[]{5, 10, 20, 50, 100, 200, 10000};
        int maxIts = 10000;
        
        for (String net : nets) {
            for (String algorithm : algNames) {
                for (String refinement : refinements) {
                    for (int nClient : nClients) {
                        for (int maxEdgesIt : maxEdgesIts) {
                            launchExperiment(net, algorithm, refinement, "BN_FusionUnion", "BN_FusionUnion", bbdd, nClient, maxEdgesIt, maxIts);
                            launchExperiment(net, algorithm, refinement, "BN_FusionIntersection", "BN_FusionIntersection", bbdd, nClient, maxEdgesIt, maxIts);
                            System.gc();
                        }
                    }
                }
            }
        }
    }

    public static void launchExperiment(String net, String algName, String refinement, String fusionC, String fusionS, String bbdd, int nClients, int maxEdgesIt, int nIterations) {
        System.out.println("\n\n\n----------------------------------------------------------------------------- \n"
                    + "Net: " + net + ", Alg Name: " + algName + ", Max. Edges It.: " + maxEdgesIt + ", Refinement: " + refinement + ", Fusion Client: " + fusionC + ", Fusion Server: " + fusionS
                            + "\n-----------------------------------------------------------------------------");
        
        String savePath = "results/temp/" + net + "-" + algName + "-" + maxEdgesIt + "-" + refinement + "-" + fusionC + "-" + fusionS + "-" + bbdd + "-" + nClients;
               
        // Create the directory if it does not exist
        File directory = new File(savePath.substring(0, savePath.lastIndexOf("/")));
        if (!directory.exists()){
            directory.mkdirs();
        }
        
        if ((!checkExistentFile(savePath))) { 
            
            DataSet allData = readData("./res/networks/BBDD/" + net + "." + bbdd + ".csv");
            ArrayList<DataSet> divisionData = divideDataSet(allData, nClients);

            ArrayList<BN_DataSet> BNDataSets = new ArrayList<>();
            for (int i = 0; i < nClients; i++) {
                BNDataSets.add(new BN_DataSet(divisionData.get(i), (net + "." + bbdd + "." + i)));
            }

            ArrayList<Client> clients = new ArrayList<>();
            for (int i = 0; i < nClients; i++) {
                Fusion fusionClient = null;
                if (fusionC.equals("BN_FusionIntersection")) {
                    fusionClient = new BN_FusionIntersection();
                } else if (fusionC.equals("BN_FusionUnion")) {
                    fusionClient = new BN_FusionUnion();
                }
                
                BNDataSets.get(i).setOriginalBNPath("./res/networks/" + net + ".xbif");
                LocalAlgorithm algorithm = new BN_GES(algName, refinement, maxEdgesIt);

                Client client = new Client(fusionClient, algorithm, BNDataSets.get(i));
                client.setStats(true);
                client.setExperimentName(algName + "," + maxEdgesIt + "," + fusionC + "," + refinement + "," + fusionS);
                clients.add(client);
            }

            Fusion fusionServer = null;
                if (fusionS.equals("BN_FusionIntersection")) {
                    fusionServer = new BN_FusionIntersection();
                } else if (fusionS.equals("BN_FusionUnion")) {
                    fusionServer = new BN_FusionUnion();
                }
            Server server = new Server(fusionServer, clients);

            server.setStats(true);
            server.setOriginalBNPath("./res/networks/" + net + ".xbif");
            server.setBBDDName(net + "." + bbdd);
            server.setExperimentName(algName + "," + maxEdgesIt + "," + fusionC + "," + refinement + "," + fusionS);
            server.setnIterations(nIterations);

            server.run();
            
            writeExistentFile(savePath);
        } else {
            System.out.println("\n EXISTENT EXPERIMENT: " + savePath + "\n");
        }
    }

    public static void launchExperiment(String net, String algName, String refinement, String fusionC, String fusionS, String[] clientBBDDs, int maxEdgesIt, int nIterations) {
        System.out.println("\n\n\n----------------------------------------------------------------------------- \n"
                    + "Net: " + net + ", Alg Name: " + algName + ", Max. Edges It.: " + maxEdgesIt + ", Refinement: " + refinement + ", Fusion Client: " + fusionC + ", Fusion Server: " + fusionS
                            + "\n-----------------------------------------------------------------------------");
        
        String savePath = "results/temp/" + net + "-" + algName + "-" + maxEdgesIt + "-" + refinement + "-" + fusionC + "-" + fusionS + "-" + clientBBDDs.length;
               
        // Create the directory if it does not exist
        File directory = new File(savePath.substring(0, savePath.lastIndexOf("/")));
        if (!directory.exists()){
            directory.mkdirs();
        }
        
        if ((!checkExistentFile(savePath))) { 
            ArrayList<Client> clients = new ArrayList<>();
            for (String bbdd : clientBBDDs) {
                Fusion fusionClient = null;
                if (fusionC.equals("BN_FusionIntersection")) {
                    fusionClient = new BN_FusionIntersection();
                } else if (fusionC.equals("BN_FusionUnion")) {
                    fusionClient = new BN_FusionUnion();
                }
                BN_DataSet data = new BN_DataSet("./res/networks/BBDD/" + net + "." + bbdd + ".csv", net + "_" + bbdd);
                data.setOriginalBNPath("./res/networks/" + net + ".xbif");
                LocalAlgorithm algorithm = new BN_GES(algName, refinement, maxEdgesIt);

                Client client = new Client(fusionClient, algorithm, data);
                client.setStats(true);
                client.setExperimentName(algName + "," + maxEdgesIt + ',' + fusionC + "," + refinement + "," + fusionS);
                clients.add(client);
            }

            Fusion fusionServer = null;
                if (fusionS.equals("BN_FusionIntersection")) {
                    fusionServer = new BN_FusionIntersection();
                } else if (fusionS.equals("BN_FusionUnion")) {
                    fusionServer = new BN_FusionUnion();
                }
            Server server = new Server(fusionServer, clients);

            server.setStats(true);
            server.setOriginalBNPath("./res/networks/" + net + ".xbif");
            server.setBBDDName(net);
            server.setExperimentName(algName + "," + maxEdgesIt + "," + fusionC + "," + refinement + "," + fusionS);
            server.setnIterations(nIterations);

            server.run();
            
            writeExistentFile(savePath);
        } else {
            System.out.println("\n EXISTENT EXPERIMENT: " + savePath + "\n");
        }
    }

    public static boolean checkExistentFile(String savePath) {
        File file = new File(savePath);
        
        return file.length() != 0;
    }
    
    public static void writeExistentFile(String savePath) {
        BufferedWriter csvWriter = null;
        try {
            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            csvWriter.append(" ");
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }
    }
}
