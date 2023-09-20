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
import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.algorithms.bnbuilders.Circular_GES;
import org.albacete.simd.algorithms.bnbuilders.Fges_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.PGESwithStages;
import org.albacete.simd.clustering.HierarchicalClustering;
import org.albacete.simd.framework.BNBuilder;
import org.bayesfl.data.BN_DataSet;
import org.bayesfl.data.Data;
import org.bayesfl.model.BN;
import org.bayesfl.model.Model;

public class BN_GES implements LocalAlgorithm {

    private BNBuilder algorithm;
    private final String algorithmName;
    private int nThreads = 4;
    private int nInterleaving = Integer.MAX_VALUE;

    public BN_GES(String algorithmName) {
        this.algorithmName = algorithmName;
    }

    /**
     * Build the local model using the algorithm.
     * @param localModel The previous local Model (BN) that the algorithm uses as base.
     * @param data The Data used to build the Model.
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        build(data);

        Graph graph = ((BN)localModel).getModel();
        algorithm.setInitialGraph(graph);

        Dag_n result = new Dag_n(algorithm.search());
        return new BN(result);
    }

    /**
     * Build the local model using the algorithm, without previous local model.
     * @param data The Data used to build the Model (BN).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Data data) {
        build(data);

        Dag_n result = new Dag_n(algorithm.search());
        return new BN(result);
    }

    private void build(Data data) {
        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        DataSet dataSet = ((BN_DataSet) data).getData();

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
    }

    public void pGES (DataSet data) {
        HierarchicalClustering clustering = new HierarchicalClustering();
        algorithm = new PGESwithStages(data, clustering, this.nThreads, Integer.MAX_VALUE, this.nInterleaving, false, true, true);
    }

    public void cGES (DataSet data) {
        HierarchicalClustering clustering = new HierarchicalClustering();
        algorithm = new Circular_GES(data, clustering, this.nThreads, this.nInterleaving, "c4");
    }

    public void fGES (DataSet data) {
        algorithm = new Fges_BNBuilder(data, false);
    }

    public void GES (DataSet data) {
        algorithm = new GES_BNBuilder(data, true);
    }

    public void setNThreads(int nThreads) {
        this.nThreads = nThreads;
    }

    public void setNInterleaving(int nInterleaving) {
        this.nInterleaving = nInterleaving;
    }
}
