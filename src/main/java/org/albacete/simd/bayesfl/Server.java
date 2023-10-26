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
 *    Server.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.bayesfl;

import org.albacete.simd.bayesfl.fusion.Fusion;
import org.albacete.simd.bayesfl.model.Model;

import java.util.Collection;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.data.Data;

public class Server {

    /**
     * The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     */
    private final Fusion globalFusion;

    /**
     * The global model build by the server.
     */
    private Model globalModel;

    /**
     * The clients that will send their local models to the server.
     */
    private final Collection<Client> clients;

    /**
     * The local models of the clients.
     */
    private final Model[] localModels;
    
    /**
     * The local models of the clients on the previous iteration.
     */
    private Model[] lastLocalModels;

    /**
     * The number of iterations of the algorithm.
     */
    private int nIterations = 5;

    /**
     * The stats flag.
     */
    private boolean stats = false;

    private String originalBNPath;
    
    private String bbddName;
            
    private String experimentName;

    /**
     * Constructor of the class Server.
     * @param globalFusion The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     * @param clients The Clients that will be run.
     */
    public Server(Fusion globalFusion, Collection<Client> clients) {
        this.globalFusion = globalFusion;
        this.clients = clients;

        int id = 0;
        for (Client client : clients) {
            client.setID(id);
            client.setNClients(clients.size());
            id++;
        }
        localModels = new Model[clients.size()];
        lastLocalModels = new Model[clients.size()];
    }

    /**
     * Constructor of the class Server with starting global model.
     * @param globalFusion The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     * @param globalModel The starting global model.
     * @param clients The Clients that will be run.
     */
    public Server(Fusion globalFusion, Model globalModel, Collection<Client> clients) {
        this(globalFusion, clients);
        this.globalModel = globalModel;
    }

    /**
     * Run the server.
     */
    public void run() {
        // For each iteration of the algorithm
        for (int iteration = 1; iteration < nIterations+1; iteration++) {
            System.out.println("\n\n\nITERATION " + iteration + "\n");
            
            // 1. Create the local model of each client with a ParallelStream
            clients.stream().forEach(Client::buildLocalModel);

            // 2. Get the local models
            for (Client client : clients) {
                lastLocalModels[client.getID()] = localModels[client.getID()];
                localModels[client.getID()] = client.getLocalModel();
            }

            // 3. Fuse the local models into a global model
            double start = System.currentTimeMillis();
            globalModel = globalFusion.fusion(localModels);
            double time = (System.currentTimeMillis() - start) / 1000;
            
            if (stats) {
                Data data = null;
                if (this.originalBNPath != null) {
                    data = new BN_DataSet(this.bbddName);
                    ((BN_DataSet)data).setOriginalBNPath(this.originalBNPath);
                }
                globalModel.saveStats(experimentName + ",server", clients.size(), -1, data, iteration, time);
            }
            
            // 4. Fuse the global model with each local model on the clients
            clients.stream().forEach(client -> client.fusion(globalModel));
            
            // 5. Check if any of the clients has changued their model
            if (checkConvergence()) break;
        }
    }
    
    private boolean checkConvergence() {
        for (int i = 0; i < localModels.length; i++) {
            if (!localModels[i].equals(lastLocalModels[i])) return false;}
        
        return true;
    }

    /**
     * Set the stats flag.
     * @param stats The stats flag.
     */
    public void setStats(boolean stats) {
        this.stats = stats;
    }

    /**
     * Sets the path of the original BN. Used only in experiments for the stats.
     * @param path Path of the original BN.
     */
    public void setOriginalBNPath(String path) {
        this.originalBNPath = path;
    }
    
    public void setBBDDName(String bbddName) {
        this.bbddName = bbddName;
    }
    
    public void setExperimentName(String experimentName) {
        this.experimentName = experimentName;
    }

    /**
     * Sets the number of iterations of the algorithm.
     * @param nIterations The number of iterations of the algorithm.
     */
    public void setnIterations(int nIterations) {
        this.nIterations = nIterations;
    }
}
