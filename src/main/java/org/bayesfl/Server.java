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

package org.bayesfl;

import org.bayesfl.fusion.Fusion;
import org.bayesfl.model.Model;

import java.util.Collection;
import org.bayesfl.model.BN;

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

    private final Model[] localModels;

    /**
     * The stats flag.
     */
    private boolean stats = false;
    private final String DASH = "------------------------------------------------------------------------";
    
    private String originalBNPath;

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
            id++;
        }
        localModels = new Model[clients.size()];
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
        // 1. Create the local model of each client with a ParallelStream
        clients.stream().forEach(Client::buildLocalModel);

        // 2. Get the local models
        for (Client client : clients) {
            localModels[client.getID()] = client.getLocalModel();
        }

        // 3. Fuse the local models into a global model
        double start = System.currentTimeMillis();
        this.fusion();
        double fusionTime = (System.currentTimeMillis() - start) / 1000;
        
        System.out.println("\nSERVER: FUSION done\n");
        if (stats) {
            System.out.println(DASH + "\n| SERVER FUSION stats:");
            System.out.println("| Fusion operator: " + globalModel.getClass().getSimpleName() + "\n|");
            System.out.println("| Build time: " + fusionTime + " s");
            globalModel.printStats();
            if (originalBNPath != null) {
                ((BN)globalModel).printOriginalBNStats(originalBNPath);
            }
            System.out.println(DASH + "\n");
        }

        // 4. Fuse the global model with each local model on the clients
        clients.stream().forEach(client -> client.fusion(globalModel));
    }

    /**
     * Perform the fusion of the local models that the clients send
     */
    protected void fusion() {
        globalModel = globalFusion.fusion(localModels);
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
}
