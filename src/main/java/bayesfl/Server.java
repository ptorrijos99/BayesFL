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

package bayesfl;

import bayesfl.convergence.Convergence;
import bayesfl.data.Data;
import bayesfl.fusion.Fusion;
import bayesfl.model.Model;

import java.util.Collection;

public class Server {

    /**
     * The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     */
    private final Fusion globalFusion;

    /**
     * The convergence criteria.
     */
    private final Convergence convergence;

    /**
     * The clients that will send their local models to the server.
     */
    private final Collection<Client> clients;

    /**
     * The global model build by the server.
     */
    private Model globalModel;

    /**
     * The local models of the clients.
     */
    private final Model[] localModels;

    /**
     * The number of iterations of the algorithm.
     */
    private int nIterations = 5;

    /**
     * The stats flag.
     */
    private boolean stats = false;

    /**
     * The data used only in experiments for the stats.
     */
    private Data data;
    
    private String path;

    private String experimentName;

    /**
     * Constructor of the class Server.
     * @param globalFusion The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     * @param convergence The convergence criteria.
     * @param clients The Clients that will be run.
     */
    public Server(Fusion globalFusion, Convergence convergence, Collection<Client> clients) {
        this.globalFusion = globalFusion;
        this.clients = clients;
        this.convergence = convergence;

        int id = 0;
        for (Client client : clients) {
            client.setID(id);
            client.setNClients(clients.size());
            id++;
        }
        localModels = new Model[clients.size()];
    }

    /**
     * Constructor of the class Server with starting global model.
     * @param globalFusion The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     * @param globalModel The starting global model.
     * @param convergence The convergence criteria.
     * @param clients The Clients that will be run.
     */
    public Server(Fusion globalFusion, Model globalModel, Convergence convergence, Collection<Client> clients) {
        this(globalFusion, convergence, clients);
        this.globalModel = globalModel;
    }

    /**
     * Run the server.
     */
    public void run() {
        // For each iteration of the algorithm
        for (int iteration = 1; iteration <= nIterations; iteration++) {
            System.out.println("\n\nITERATION " + iteration + "\n");

            // 1. Create the local model of each client with a ParallelStream
            clients.parallelStream().forEach(Client::buildLocalModel);

            // 2 Apply the differential privacy to the local models
            clients.parallelStream().forEach(Client::applyDifferentialPrivacy);

            // 3. Get the local models
            for (Client client : clients) {
                localModels[client.getID()] = client.getLocalModel();
            }

            // 4. Check if any of the clients has changed their model
            if (convergence.checkConvergence(localModels)) break;

            // 5. Fuse the local models into a global model
            double start = System.currentTimeMillis();
            globalModel = globalFusion.fusion(localModels);
            double time = (System.currentTimeMillis() - start) / 1000;
            
            if (stats) {
                globalModel.saveStats(experimentName, "Server", path, clients.size(), -1, data, iteration, time);
            }

            // 6. Fuse the global model with each local model on the clients
            clients.parallelStream().forEach(client -> client.fusion(globalModel));
        }
    }

    /**
     * Set the stats flag.
     * @param stats The stats flag.
     */
    public void setStats(boolean stats, String path) {
        this.stats = stats;
        this.path = path;
    }

    /**
     * Sets the data used only in experiments for the stats.
     * @param data The data used only in experiments for the stats.
     */
    public void setData(Data data) {
        this.data = data;
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

    /**
     * Get the global model of the server.
     * @return The global model of the server.
     */
    public Model getGlobalModel() {
        return globalModel;
    }
}
