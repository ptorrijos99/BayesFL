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

package org.albacete.simd.bayesfl;

import org.albacete.simd.bayesfl.algorithms.BN_GES;
import org.albacete.simd.bayesfl.algorithms.LocalAlgorithm;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.fusion.BN_FusionUnion;
import org.albacete.simd.bayesfl.fusion.BN_FusionIntersection;
import org.albacete.simd.bayesfl.fusion.Fusion;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        String net = "andes";
        String algName = "pGES";

        ArrayList<Client> clients = new ArrayList<>();
        for (int i = 1; i < 9; i++) {
            Fusion fusionClient = new BN_FusionUnion();
            BN_DataSet data = new BN_DataSet("./res/networks/BBDD/" + net + ".xbif5000" + i + "_.csv", net + "_5000" + i);
            data.setOriginalBNPath("./res/networks/" + net + ".xbif");
            LocalAlgorithm algorithm = new BN_GES(algName);
            
            Client client = new Client(fusionClient, algorithm, data);
            client.setStats(true);
            clients.add(client);
        }

        Fusion fusionServer = new BN_FusionUnion();
        Server server = new Server(fusionServer, clients);
        
        server.setStats(true);
        server.setOriginalBNPath("./res/networks/" + net + ".xbif");
        server.setBBDDName(net);
        server.setExperimentName(algName);
        server.setnIterations(4);
        
        server.run();
    }
}
