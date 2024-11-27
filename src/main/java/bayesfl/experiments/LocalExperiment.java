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

package bayesfl.experiments;

import bayesfl.algorithms.BN_GES;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.convergence.Convergence;
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

public class LocalExperiment {
    public static String PATH = "./";
    
    public static void main(String[] args) {
        simpleExperiment();
        //multipleExperiment();
    }
    
    public static void simpleExperiment() {
        String net = "alarm";
        String algName = "GES";
        String refinement = "None";
        String fusionClient = "Union";
        String fusionServer = "MaxFrequency";//"GeneticTW" "MaxTreewidth" "MaxFrequency" "MinCut"
        
        int maxEdgesIt = 10;
        int nIterations = 100;

        //int maxEdgesIt = 10000000;
        //int nIterations = 1;

        //String[] bbdd_paths = new String[]{"0", "1", "2", "3"};
        //launchExperiment(net, algName, refinement, fusionClient, fusionServer, bbdd_paths, maxEdgesIt, nIterations);
        
        String bbdd = "0";
        int nClients = 10;

        int percentaje = 25;
        int limitPerct = (int) Math.round(nClients * percentaje / 100.0);

        String limitC = "4";
        String limitS = "" + limitPerct;

        launchExperiment(net, algName, refinement, fusionClient, limitC, fusionServer, limitS, bbdd, nClients, maxEdgesIt, nIterations);
    }


    public static void launchExperiment(String net, String algName, String refinement, String fusionC, String limitC, String fusionS, String limitS, String bbdd, int nClients, int maxEdgesIt, int nIterations) {
        String operation = algName + "," + maxEdgesIt + "," + fusionC + "," + limitC + "," + refinement + "," + fusionS + "," + limitS;
        String savePath = "./results/Server/" + net + "." + bbdd + "_" + operation + "_" + nClients + "_-1.csv";

        if ((!checkExistentFile(savePath))) {
            
            DataSet allData = Utils.readData(PATH + "res/networks/BBDD/" + net + "." + bbdd + ".csv");
            ArrayList<DataSet> divisionData = BN_DataSet.divideDataSet(allData, nClients);

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
                LocalAlgorithm algorithm = new BN_GES(algName, refinement, maxEdgesIt);

                Client client = new Client(fusionClient, algorithm, data);
                client.setStats(true, true, PATH);
                client.setExperimentName(algName + "," + maxEdgesIt + "," + fusionC + "," + limitC + "," + refinement + "," + fusionS + "," + limitS);
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

            Convergence convergence = new ModelEquality();
            Server server = new Server(fusionServer, convergence, clients);

            server.setStats(true, PATH);
            BN_DataSet data = new BN_DataSet(net + "." + bbdd);
            data.setOriginalBNPath(PATH + "res/networks/" + net + ".xbif");
            server.setData(data);

            server.setExperimentName(algName + "," + maxEdgesIt + "," + fusionC + "," + limitC + "," + refinement + "," + fusionS + "," + limitS);
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
