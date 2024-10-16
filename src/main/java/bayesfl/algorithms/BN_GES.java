/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2024 Universidad de Castilla-La Mancha, España
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
 *    BN_GES.java
 *    Copyright (C) 2024 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.algorithms;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.algorithms.bnbuilders.Circular_GES;
import org.albacete.simd.algorithms.bnbuilders.Fges_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.PGESwithStages;
import org.albacete.simd.clustering.HierarchicalClustering;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.threads.BESThread;
import org.albacete.simd.threads.FESThread;
import bayesfl.data.BN_DataSet;
import bayesfl.data.Data;
import bayesfl.model.BN;
import bayesfl.model.Model;

import java.util.HashSet;
import java.util.Set;

public class BN_GES implements LocalAlgorithm {

    private BNBuilder algorithm;
    private String algorithmName = "GES";
    private String refinement = "None";
    private int nGESThreads = 4;
    private int nInterleaving = Integer.MAX_VALUE;

    public BN_GES() {}

    public BN_GES(String algorithmName, String refinement) {
        this.algorithmName = algorithmName;
        this.refinement = refinement;
    }

    public BN_GES(String algorithmName, String refinement, int nInterleaving) {
        this(algorithmName, refinement);
        this.nInterleaving = nInterleaving;
    }

    /**
     * Build the local model using the algorithm, without previous local model.
     * @param data The Data (BN_DataSet) used to build the Model (BN).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Data data) {
        return buildLocalModel(null, data);
    }

    /**
     * Build the local model using the algorithm.
     * @param localModel The previous local Model (BN) that the algorithm uses as base.
     * @param data The Data (BN_DataSet) used to build the Model (BN).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        DataSet dataSet = ((BN_DataSet) data).getData();

        // Initialize the algorithm
        switch (algorithmName) {
            case "pGES":
                pGES(dataSet);
                break;
            case "cGES":
                cGES(dataSet);
                break;
            case "fGES":
                fGES(dataSet);
                break;
            default:
                System.out.println("Algorithm " + algorithmName + " not found, using GES");
            case "GES":
                GES(dataSet);
                break;
        }
        
        algorithm.setnItInterleaving(nInterleaving);

        /* If there is a previous local model, use it as base. If is null (for example, with a call of
           "public Model buildLocalModel(Data data)"), the model isn't an instance of BN. */
        if (localModel instanceof BN bn) {
            algorithm.setInitialGraph(bn.getModel());
        }

        // Search with the algorithm created
        return new BN(algorithm.search());
    }

    private void pGES(DataSet data) {
        HierarchicalClustering clustering = new HierarchicalClustering();
        algorithm = new PGESwithStages(data, clustering, nGESThreads, Integer.MAX_VALUE, nInterleaving, false, true, true);
    }

    private void cGES(DataSet data) {
        HierarchicalClustering clustering = new HierarchicalClustering();
        algorithm = new Circular_GES(data, clustering, nGESThreads, nInterleaving, "c4");
    }

    private void fGES(DataSet data) {
        algorithm = new Fges_BNBuilder(data, false);
    }

    private void GES(DataSet data) {
        algorithm = new GES_BNBuilder(data, true);
    }

    /**
     * Refinate the local model using the algorithm.
     * @param oldModel The previous local Model that the algorithm refines.
     * @param localModel The local Model from witch the algorithm get the changes to do the refinement.
     * @param data The Data used to build the Model.
     * @return The refined model build by the algorithm.
     */
    @Override
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        Graph oldM = ((BN)oldModel).getModel();
        Graph localM = ((BN)localModel).getModel();
        DataSet dataSet = ((BN_DataSet) data).getData();

        switch (refinement) {
            case "FES" -> localModel = new BN(refinementFES(oldM, localM, dataSet));
            case "BES" -> localModel = new BN(refinementBES(oldM, localM, dataSet));
            case "GES" -> {
                Graph refinedFES = refinementFES(oldM, localM, dataSet);
                Graph refinedBES = refinementBES(refinedFES, localM, dataSet);
                localModel = new BN(refinedBES);
            }
            default -> {
            }
        }

        return localModel;
    }

    /**
     * Refinate the local model using the FES algorithm.
     * @param oldModel The previous local Model that the algorithm refines.
     * @param localModel The local Model from witch the algorithm get the changes to do the refinement (add edges).
     * @param data The Data used to build the Model.
     * @return The refined model build by the algorithm.
     */
    private Graph refinementFES(Graph oldModel, Graph localModel, DataSet data) {
        Set<Edge> candidates = getEdgesDifferences(oldModel, localModel);

        FESThread fes = new FESThread(algorithm.getProblem(), oldModel, candidates, candidates.size(), false, true,true);
        algorithm.getProblem().setData(data);
        fes.run();
        try {
            localModel = fes.getCurrentGraph();
        } catch (InterruptedException ignored) {}

        return localModel;
    }

    /**
     * Refinate the local model using the FES algorithm.
     * @param oldModel The previous local Model that the algorithm refines.
     * @param localModel The local Model from witch the algorithm get the changes to do the refinement (remove edges).
     * @param data The Data used to build the Model.
     * @return The refined model build by the algorithm.
     */
    private Graph refinementBES(Graph oldModel, Graph localModel, DataSet data) {
        Set<Edge> candidates = getEdgesDifferences(localModel, oldModel);

        BESThread bes = new BESThread(algorithm.getProblem(), oldModel, candidates);
        algorithm.getProblem().setData(data);
        bes.run();
        try {
            localModel = bes.getCurrentGraph();
        } catch (InterruptedException ignored) {}

        return localModel;
    }

    /**
     * Get the edges that are in the model2 but not in the model1.
     * @param model1 The first model.
     * @param model2 The second model.
     * @return The edges that are in the model2 but not in the model1.
     */
    private Set<Edge> getEdgesDifferences(Graph model1, Graph model2) {
        Set<Edge> candidates = new HashSet<>();

        for (Edge e : model2.getEdges()) {
            if (model1.getEdge(e.getNode1(), e.getNode2()) != null ||
                    model1.getEdge(e.getNode2(), e.getNode1()) != null)
                continue;
            candidates.add(Edges.directedEdge(e.getNode1(), e.getNode2()));
            candidates.add(Edges.directedEdge(e.getNode2(), e.getNode1()));
        }

        return candidates;
    }

    public void setNGESThreads(int nGESThreads) {
        this.nGESThreads = nGESThreads;
    }

    public void setNInterleaving(int nInterleaving) {
        this.nInterleaving = nInterleaving;
    }

    @Override
    public String getAlgorithmName() {
        return algorithmName;
    }
    
    @Override
    public String getRefinementName() {
        return refinement;
    }
}
