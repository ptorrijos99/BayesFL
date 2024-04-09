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
 *    MCT_MCTS.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.algorithms;

import edu.cmu.tetrad.graph.Dag;
import bayesfl.data.BN_DataSet;
import bayesfl.data.Data;
import bayesfl.model.BN;
import bayesfl.model.MCT;
import bayesfl.model.Model;
import org.albacete.simd.mctsbn.MCTSBN;
import org.albacete.simd.mctsbn.TreeNode;

import static bayesfl.experiments.ExperimentUtils.calculateBDeu;

public class MCT_MCTS implements LocalAlgorithm {

    private int limitIteration = 10;
    private double exploitation = 50;
    private double probabilitySwap = 0.25;
    private double numberSwaps = 0.2;
    private String initializeAlgorithm = "GES";

    private MCTSBN algorithm;

    public MCT_MCTS() {}

    public MCT_MCTS(int limitIteration, double exploitation, double probabilitySwap, double numberSwaps, String initializeAlgorithm) {
        this.limitIteration = limitIteration;
        this.exploitation = exploitation;
        this.probabilitySwap = probabilitySwap;
        this.numberSwaps = numberSwaps;
        this.initializeAlgorithm = initializeAlgorithm;
    }

    /**
     * Build the local model using the algorithm, without previous local model.
     *
     * @param data The Data (BN_DataSet) used to build the Model (MCT).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Data data) {
        return buildLocalModel(null, data);
    }

    /**
     * Build the local model using the algorithm.
     *
     * @param localModel The previous local Model (MCT) that the algorithm uses as base.
     * @param data       The Data (BN_DataSet) used to build the Model (MCT).
     * @return The model build by the algorithm.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        if (algorithm == null) {
            algorithm = new MCTSBN(((BN_DataSet) data).getProblem(), limitIteration, exploitation, probabilitySwap, numberSwaps, initializeAlgorithm);
        }

        BN lastModel = null;

        /* If there is a previous local model, use it as base. If is null (for example, with a call of
           "public Model buildLocalModel(Data data)"), the model isn't an instance of MCT. */
        if (localModel instanceof MCT mct) {
            algorithm.setInitialTree((TreeNode) mct.getModel());
            System.out.println(" BDeu inicial: " + calculateBDeu(data, lastModel.getModel()));
            mct.calculateBestBN(data);
            lastModel = mct.getBestBN();
            algorithm.setBestDag(lastModel.getModel(), lastModel.getScore());
            System.out.println(" BDeu después BestBN: " + calculateBDeu(data, lastModel.getModel()));
        }

        // Search with the algorithm created
        Dag bestBN = algorithm.search();
        System.out.println(" BDeu final: " + calculateBDeu(data, bestBN) + "\n");
        return new MCT(algorithm.getTreeRoot(), new BN(bestBN), lastModel);
    }

    /**
     * Refinement is not implemented in this algorithm.
     */
    @Override
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return null;
    }

    @Override
    public String getAlgorithmName() {
        return "MCTS";
    }

    /**
     * Refinement is not implemented in this algorithm.
     */
    @Override
    public String getRefinementName() {
        return "None";
    }
}
