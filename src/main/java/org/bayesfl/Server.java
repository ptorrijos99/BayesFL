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

import java.util.Collection;

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
     * Constructor of the class Server.
     * @param globalFusion The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     */
    public Server(Fusion globalFusion, Collection<Client> clients) {
        this.globalFusion = globalFusion;
        this.clients = clients;

        int id = 0;
        for (Client client : clients) {
            client.setServer(this);
            client.setID(id);
            id++;
        }
    }

    /**
     * Constructor of the class Server with starting global model.
     * @param globalFusion The Fusion operator that will be used to perform the fusion of the local models that the clients
     * send with the global model of the server.
     * @param globalModel The starting global model.
     */
    public Server(Fusion globalFusion, Model globalModel, Collection<Client> clients) {
        this(globalFusion, clients);
        this.globalModel = globalModel;
    }

    /**
     * Run the server.
     */
    public void run() {
        // 1. Run the clients
    }
}
