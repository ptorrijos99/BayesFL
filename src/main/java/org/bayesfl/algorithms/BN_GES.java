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
 *    BN_pGES.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.bayesfl.algorithms;

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
import org.bayesfl.data.BN_DataSet;
import org.bayesfl.data.Data;
import org.bayesfl.model.BN;
import org.bayesfl.model.Model;

import java.util.HashSet;
import java.util.Set;

public class BN_GES implements LocalAlgorithm {

    private BNBuilder algorithm;
    private final String algorithmName;
    private String dataName;
    private String refinement = "None";
    private int nGESThreads = 4;
    private int nInterleaving = Integer.MAX_VALUE;
    private double buildTime;
    private double refinementTime;



    public BN_GES(String algorithmName) {
        this.algorithmName = algorithmName;
    }

    public BN_GES(String algorithmName, String refinement) {
        this(algorithmName);
        this.refinement = refinement;
    }

    /**
     * Build the local model using the algorithm, without previous local model.
     * @param data The Data used to build the Model (BN).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Data data) {
        return buildLocalModel(null, data);
    }

    /**
     * Build the local model using the algorithm.
     * @param localModel The previous local Model (BN) that the algorithm uses as base.
     * @param data The Data used to build the Model.
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }
        double startTime = System.currentTimeMillis();
        DataSet dataSet = ((BN_DataSet) data).getData();
        dataName = data.getName();

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

        /* If there is a previous local model, use it as base. If is null (for example, with a call of
           "public Model buildLocalModel(Data data)"), the model isn't an instance of BN. */
        if (localModel instanceof BN bn) {
            Graph graph = bn.getModel();
            algorithm.setInitialGraph(graph);
        }

        // Search with the algorithm created
        BN result = new BN(algorithm.search());
        buildTime = (System.currentTimeMillis() - startTime) / 1000;

        return result;
    }

    private void pGES (DataSet data) {
        HierarchicalClustering clustering = new HierarchicalClustering();
        algorithm = new PGESwithStages(data, clustering, this.nGESThreads, Integer.MAX_VALUE, this.nInterleaving, false, true, true);
    }

    private void cGES (DataSet data) {
        HierarchicalClustering clustering = new HierarchicalClustering();
        algorithm = new Circular_GES(data, clustering, this.nGESThreads, this.nInterleaving, "c4");
    }

    private void fGES (DataSet data) {
        algorithm = new Fges_BNBuilder(data, false);
    }

    private void GES (DataSet data) {
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

        double startTime = System.currentTimeMillis();

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

        refinementTime = (System.currentTimeMillis() - startTime) / 1000;

        return localModel;
    }

    /**
     * Refinate the local model using the FES algorithm.
     * @param oldModel The previous local Model that the algorithm refines.
     * @param localModel The local Model from witch the algorithm get the changes to do the refinement (add edges).
     * @param data The Data used to build the Model.
     * @return The refined model build by the algorithm.
     */
    private Graph refinementFES (Graph oldModel, Graph localModel, DataSet data) {
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
    private Graph refinementBES (Graph oldModel, Graph localModel, DataSet data) {
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

    /**
     * Print the stats of the algorithm.
     */
    @Override
    public void printStats() {
        System.out.println(this);
        System.out.println("| Build time: " + buildTime + " s");
    }

    /**
     * Print the stats of the refinement with the algorithm.
     */
    @Override
    public void printRefinementStats() {
        System.out.println("| " + refinement + " Refinement\n|");
        System.out.println("| Refinement time: " + refinementTime + " s");
    }


    /**
     * Save the results of the algorithm.
     * @param path The path where the results are saved.
     */
    @Override
    public void saveResults(String path) {

    }

    @Override
    public String toString() {
        String string = "| " + algorithmName + " Algorithm.  " +
                "Database: " + dataName + ", Threads: " + Runtime.getRuntime().availableProcessors();
        if (algorithmName.equals("pGES") || algorithmName.equals("cGES")) {
            string = string + ", GESThreads: " + nGESThreads + ", Interleaving: " + nInterleaving;
        }
        return string + "\n|";
    }

    public void setNGESThreads(int nGESThreads) {
        this.nGESThreads = nGESThreads;
    }

    public void setNInterleaving(int nInterleaving) {
        this.nInterleaving = nInterleaving;
    }


}
