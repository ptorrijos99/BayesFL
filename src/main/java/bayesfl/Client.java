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
 *    Client.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl;

import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.data.Data;
import bayesfl.fusion.Fusion;
import bayesfl.model.Model;
import bayesfl.privacy.DenoisableModel;
import bayesfl.privacy.NoiseGenerator;

public class Client {

    /**
     * The Fusion operator that will be used to perform the fusion of the global model that the server
     * sends with the local model of the client
     */
    private final Fusion localFusion;

    /**
     * The local algorithm used to build the local model.
     */
    private final LocalAlgorithm localAlgorithm;

    /**
     * The data that the client will use to build the local model.
     */
    private final Data data;

    /**
     * The local model of the client.
     */
    private Model localModel;

    /**
     * The noise generator used to apply the differential privacy.
     */
    private NoiseGenerator noiseGenerator = null;

    /**
     * The ID of the client.
     */
    private int id;

    /**
     * The number of clients.
     */
    private int nClients;

    /**
     * The build stats flag.
     */
    private boolean buildStats = false;

    /**
     * The fusion stats flag.
     */
    private boolean fusionStats = false;
    
    private String path;

    private int iteration;
    
    private String experimentName;

    /**
     * Constructor of the class Client.
     * @param localFusion The Fusion operator that will be used to perform the fusion of the global model that the server
     * sends with the local model of the client
     * @param localAlgorithm The local algorithm used to build the local model.
     * @param data The data that the client will use to build the local model.
     */
    public Client(Fusion localFusion, LocalAlgorithm localAlgorithm, Data data) {
        this.localFusion = localFusion;
        this.localAlgorithm = localAlgorithm;
        this.data = data;
        this.iteration = 0;
    }

    /**
     * Constructor of the class Client with noise generator.
     * @param localFusion The Fusion operator that will be used to perform the fusion of the global model that the server
     * sends with the local model of the client
     * @param localAlgorithm The local algorithm used to build the local model.
     * @param data The data that the client will use to build the local model.
     * @param noiseGenerator The noise generator used to apply the differential privacy.
     */
    public Client(Fusion localFusion, LocalAlgorithm localAlgorithm, Data data, NoiseGenerator noiseGenerator) {
        this(localFusion, localAlgorithm, data);
        this.noiseGenerator = noiseGenerator;
    }

    /**
     * Constructor of the class Client with starting local model.
     * @param localFusion The Fusion operator that will be used to perform the fusion of the global model that the server
     * sends with the local model of the client
     * @param localModel The local model of the client.
     * @param localAlgorithm The local algorithm used to build the local model.
     * @param data The data that the client will use to build the local model.
     */
    public Client(Fusion localFusion, Model localModel, LocalAlgorithm localAlgorithm, Data data) {
        this(localFusion, localAlgorithm, data);
        this.localModel = localModel;
    }

    /**
     * Constructor of the class Client with starting local model and noise generator.
     * @param localFusion The Fusion operator that will be used to perform the fusion of the global model that the server
     * sends with the local model of the client
     * @param localModel The local model of the client.
     * @param localAlgorithm The local algorithm used to build the local model.
     * @param data The data that the client will use to build the local model.
     * @param noiseGenerator The noise generator used to apply the differential privacy.
     */
    public Client(Fusion localFusion, Model localModel, LocalAlgorithm localAlgorithm, Data data, NoiseGenerator noiseGenerator) {
        this(localFusion, localModel, localAlgorithm, data);
        this.noiseGenerator = noiseGenerator;
    }

    /**
     * Build the local model of the client.
     */
    protected void buildLocalModel() {
        iteration = iteration+1;
        
        double start = System.currentTimeMillis();
        localModel = localAlgorithm.buildLocalModel(localModel, data);
        double time = (System.currentTimeMillis() - start) / 1000;
        
        if (buildStats)  {
            localModel.saveStats(experimentName, "Client/Build", path, nClients, id, data, iteration, time);
        }
    }

    /**
     * Apply the differential privacy to the local model if defined.
     */
    protected void applyDifferentialPrivacy() {
        if (noiseGenerator == null || !(localModel instanceof DenoisableModel dm)) return;
        dm.applyNoise(noiseGenerator);
    }

    /**
     * Perform the fusion of the global model that the server
     * sends with the local model of the client.
     * Then, if defined, a refinement of the local model is performed.
     * @param globalModel The global model that the server sends.
     */
    protected void fusion(Model globalModel) {
        Model oldModel = localModel;

        // Perform the fusion between the local and the global model
        double start = System.currentTimeMillis();
        localModel = localFusion.fusion(localModel, globalModel);
        double time = (System.currentTimeMillis() - start) / 1000;

        if (fusionStats) {
            localModel.saveStats(experimentName, "Client/Fusion", path, nClients, id, data, iteration, time);
        }
        
        // If defined, perform a refinement to the local model
        if (!localAlgorithm.getRefinementName().equals("None")) {
            start = System.currentTimeMillis();
            localModel = localAlgorithm.refinateLocalModel(oldModel, localModel, data);
            time = (System.currentTimeMillis() - start) / 1000;

            if (fusionStats)  {
                localModel.saveStats(experimentName, "Client/Refinement", path, nClients, id, data, iteration, time);
            }
        }
    }

    /**
     * Set the ID of the client.
     * @param id The ID of the client.
     */
    public void setID(int id) {
        this.id = id;
    }

    /**
     * Set the number of clients.
     * @param nClients The number of clients.
     */
    public void setNClients(int nClients) {
        this.nClients = nClients;
    }

    /**
     * Get the ID of the client.
     * @return The ID of the client.
     */
    public int getID() {
        return this.id;
    }

    /**
     * Get the local model of the client.
     * @return The local Model of the client.
     */
    public Model getLocalModel() {
        return localModel;
    }

    /**
     * Get the local algorithm used to build the local model.
     * 
     * @return The local algorithm used to build the local model.
     */
    public LocalAlgorithm getLocalAlgorithm() {
        return this.localAlgorithm;
    }

    /**
     * Set the stats flag.
     * @param buildStats The stats flag.
     * @param fusionStats The stats flag.
     */
    public void setStats(boolean buildStats, boolean fusionStats, String path) {
        this.buildStats = buildStats;
        this.fusionStats = fusionStats;
        this.path = path;
    }

    public void setExperimentName(String experimentName) {
        this.experimentName = experimentName;
    }
}
