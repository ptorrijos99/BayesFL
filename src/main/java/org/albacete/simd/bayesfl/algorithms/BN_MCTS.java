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
 *    BN_MCTS.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.bayesfl.algorithms;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.data.Data;
import org.albacete.simd.bayesfl.model.BN;
import org.albacete.simd.bayesfl.model.Model;
import org.albacete.simd.mctsbn.MCTSBN;

public class BN_MCTS implements LocalAlgorithm {

    private MCTSBN algorithm;
    private final String initializeAlgorithmName;
    private String dataName;

    /**
     * Constructor of the class.
     * @param initializeAlgorithmName The name of the algorithm used to initialize the local model.
     */
    public BN_MCTS(String initializeAlgorithmName) {
        this.initializeAlgorithmName = initializeAlgorithmName;
    }

    /**
     * Build the local model using the algorithm, without previous local model.
     *
     * @param data The Data (BN_DataSet) used to build the Model (BN).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Data data) {
        return buildLocalModel(null, data);
    }

    /**
     * Build the local model using the algorithm.
     *
     * @param localModel The previous local Model (BN) that the algorithm uses as base.
     * @param data       The Data (BN_DataSet) used to build the Model (BN).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        DataSet dataSet = ((BN_DataSet) data).getData();
        dataName = data.getName();





        // Search with the algorithm created
        BN result = new BN(algorithm.search());

        return result;
    }

    /**
     * Refinate the local model using the algorithm.
     *
     * @param oldModel   The previous local Model that the algorithm refines.
     * @param localModel The local Model from witch the algorithm get the changes to do the refinement.
     * @param data       The Data used to build the Model.
     * @return The refined model build by the algorithm.
     */
    @Override
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return null;
    }

    @Override
    public String getAlgorithmName() {
        return this.initializeAlgorithmName;
    }

    @Override
    public String getRefinementName() {
        return null;
    }
}
