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
 *    LocalMCTSExperiment.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.bayesfl.experiments;

import edu.cmu.tetrad.data.DataSet;
import org.albacete.simd.bayesfl.Client;
import org.albacete.simd.bayesfl.Server;
import org.albacete.simd.bayesfl.algorithms.LocalAlgorithm;
import org.albacete.simd.bayesfl.algorithms.MCT_MCTS;
import org.albacete.simd.bayesfl.convergence.Convergence;
import org.albacete.simd.bayesfl.convergence.NoneConvergence;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.fusion.*;

import java.util.ArrayList;

import static org.albacete.simd.bayesfl.data.BN_DataSet.divideDataSet;
import static org.albacete.simd.bayesfl.data.BN_DataSet.readData;

public class LocalMCTSExperiment {
    public static String PATH = "./";
    
    public static void main(String[] args) {
        simpleExperiment();
    }
    
    public static void simpleExperiment() {
        String net = "andes";
        String algName = "MCTS";
        String intitialAlgorithm = "pGES";

        int maxEdgesIt = 50;
        int nIterations = 10 ;

        int exploitation = 10;
        double probability_swap = 0.2;
        double number_swaps = 0.2;

        String bbdd = "0";
        int nClients = 2;
        launchExperiment(net, algName, bbdd, nClients, maxEdgesIt, nIterations, exploitation, probability_swap, number_swaps, intitialAlgorithm);
    }

    public static void launchExperiment(String net, String algName, String bbdd, int nClients, int maxEdgesIt, int nIterations, int exploitation, double probability_swap, double number_swaps, String initializeAlgorithm) {

        String operation = algName + "," + maxEdgesIt + ",server";
        String savePath = "./results/Server/" + net + "." + bbdd + "_" + operation + "_" + nClients + "_-1.csv";

        //if ((!LocalExperiment.checkExistentFile(savePath))) {
            DataSet allData = readData(PATH + "res/networks/BBDD/" + net + "." + bbdd + ".csv");
            ArrayList<DataSet> divisionData = divideDataSet(allData, nClients);

            ArrayList<Client> clients = new ArrayList<>();
            for (int i = 0; i < nClients; i++) {
                Fusion fusionClient = new FusionPosition(-1);

                BN_DataSet data = new BN_DataSet(divisionData.get(i), (net + "." + bbdd + "." + i));
                data.setOriginalBNPath(PATH + "res/networks/" + net + ".xbif");
                LocalAlgorithm algorithm = new MCT_MCTS(maxEdgesIt, exploitation, probability_swap, number_swaps, initializeAlgorithm);

                Client client = new Client(fusionClient, algorithm, data);
                client.setStats(true, true, PATH);
                client.setExperimentName(algName + "," + maxEdgesIt);
                clients.add(client);
            }

            Fusion fusionServer = new MCT_Fusion();
            Convergence convergence = new NoneConvergence();
            Server server = new Server(fusionServer, convergence, clients);

            server.setStats(true, PATH);
            BN_DataSet data = new BN_DataSet(net + "." + bbdd);
            data.setOriginalBNPath(PATH + "res/networks/" + net + ".xbif");
            server.setData(data);

            server.setExperimentName(algName + "," + maxEdgesIt);
            server.setnIterations(nIterations);

            server.run();
            
            LocalExperiment.writeExistentFile(savePath);
        //} else {
        //    System.out.println("\n EXISTENT EXPERIMENT: " + savePath + "\n");
        //}
    }

}
