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
 *    Main.java
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
import org.albacete.simd.bayesfl.algorithms.BN_GES;
import org.albacete.simd.bayesfl.algorithms.LocalAlgorithm;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.fusion.BN_FusionUnion;
import org.albacete.simd.bayesfl.fusion.BN_FusionIntersection;
import org.albacete.simd.bayesfl.fusion.Fusion;

import java.util.ArrayList;
import org.albacete.simd.bayesfl.Client;
import org.albacete.simd.bayesfl.Server;

public class LocalExperiment {
    public static void main(String[] args) {
        //simpleExperiment();
        multipleExperiment();
    }
    
    public static void simpleExperiment() {
        String net = "alarm";
        String algName = "pGES";
        String refinement = "GES";
        String fusionClient = "BN_FusionUnion";
        String fusionServer = "BN_FusionIntersection";

        launchExperiment(net, algName, refinement, fusionClient,fusionServer, 5, 5);
    }
    
    public static void multipleExperiment() {
        String[] nets = new String[]{"alarm", "barley", "child", "hailfinder", "hepar2", "insurance", "mildew", "water", "win95pts", "andes", "diabetes", "link", "pigs", "pathfinder", "munin"};
        //String[] bbdd_paths = new String[]{"50001", "50002", "50003", "50004", "50005", "50006", "50007", "50008", "50009", "50001246", ""};
        String[] bbdd_paths = new String[]{"50001", "50002", "50003", "50004"};
        String[] algNames = new String[]{"pGES", "GES"};
        String[] refinements = new String[]{"None", "GES"};
        String[] fusionsClient = new String[]{"BN_FusionUnion", "BN_FusionIntersection"};
        String[] fusionsServer = new String[]{"BN_FusionUnion", "BN_FusionIntersection"};
        
        for (String net : nets) {
            for (String algorithm : algNames) {
                for (String refinement : refinements) {
                    for (String fusionClient : fusionsClient) {
                        for (String fusionServer : fusionsServer) {
                            try {
                                launchExperiment(net, algorithm, refinement, fusionClient,fusionServer, bbdd_paths,5);
                            } catch(IOException ex) {System.out.println(ex);}
                            System.gc();
                        }
                    }
                }
            }
        }
    }
    
    public static void launchExperiment(String net, String algName, String refinement, String fusionC, String fusionS, String[] clientBBDDs, int nIterations) throws IOException {
        System.out.println("\n\n\n----------------------------------------------------------------------------- \n"
                    + "Net: " + net + ", Alg Name: " + algName + ", Refinement: " + refinement + ", Fusion Client: " + fusionC + ", Fusion Server: " + fusionS
                            + "\n-----------------------------------------------------------------------------");
        
        String savePath = "results/temp/" + net + "-" + algName + "-" + refinement + "-" + fusionC + "-" + fusionS;
               
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
                BN_DataSet data = new BN_DataSet("./res/networks/BBDD/" + net + ".xbif" + bbdd + "_.csv", net + "_" + bbdd);
                data.setOriginalBNPath("./res/networks/" + net + ".xbif");
                LocalAlgorithm algorithm = new BN_GES(algName, refinement);

                Client client = new Client(fusionClient, algorithm, data);
                client.setStats(true);
                client.setExperimentName(algName + "_" + fusionC + "_" + refinement + "_" + fusionS);
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
            server.setExperimentName(algName + "_" + fusionC + "_" + refinement + "_" + fusionS);
            server.setnIterations(nIterations);

            server.run();
            
            writeExistentFile(savePath);
        } else {
            System.out.println("\n EXPERIMENTO YA EXISTENTE: " + savePath + "\n");
        }
    }
    
    
    public static void launchExperiment(String net, String algName, String refinement, String fusionC, String fusionS, int nExecs, int nIterations) {
        ArrayList<Client> clients = new ArrayList<>();
        for (int i = 1; i < nExecs; i++) {
            Fusion fusionClient = null;
            if (fusionC.equals("BN_FusionIntersection")) {
                fusionClient = new BN_FusionIntersection();
            } else if (fusionC.equals("BN_FusionUnion")) {
                fusionClient = new BN_FusionUnion();
            }
            BN_DataSet data = new BN_DataSet("./res/networks/BBDD/" + net + ".xbif5000" + i + "_.csv", net + "_5000" + i);
            data.setOriginalBNPath("./res/networks/" + net + ".xbif");
            LocalAlgorithm algorithm = new BN_GES(algName, refinement);
            
            Client client = new Client(fusionClient, algorithm, data);
            client.setStats(true);
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
        server.setExperimentName(algName + "_" + fusionS + "_" + refinement);
        server.setnIterations(nIterations);
        
        server.run();
    }
    
    public static boolean checkExistentFile(String savePath) throws IOException {
        File file = new File(savePath);
        
        return file.length() != 0;
    }
    
    public static void writeExistentFile(String savePath) throws IOException {
        BufferedWriter csvWriter = new BufferedWriter(new FileWriter(savePath, true));
        csvWriter.append(" ");
        csvWriter.flush();
        csvWriter.close();
    }
}
