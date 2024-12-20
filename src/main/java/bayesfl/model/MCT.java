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
 *    MCT.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.model;

import bayesfl.experiments.utils.ExperimentUtils;
import edu.cmu.tetrad.graph.Dag;
import bayesfl.data.Data;
import org.albacete.simd.mctsbn.TreeNode;

import java.util.ArrayList;

public class MCT implements Model {

    private TreeNode treeRoot;

    private BN bestBN;

    private BN lastBN;

    private ArrayList<BN> candidates = new ArrayList<>();

    public MCT(TreeNode treeRoot, BN bestBN) {
        this.treeRoot = treeRoot;
        this.bestBN = bestBN;
    }

    public MCT(TreeNode treeRoot, BN bestBN, BN lastBN) {
        this(treeRoot, bestBN);
        this.lastBN = lastBN;
    }

    /**
     * Constructor for the MCT class. It receives the tree root and the candidates to be evaluated.
     * To be used to create the fused MCT. Then, clients try the best BNs of the other clients.
     * @param treeRoot The root of the tree.
     * @param candidates The candidate BNs to be evaluated.
     */
    public MCT(TreeNode treeRoot, ArrayList<BN> candidates) {
        this.treeRoot = treeRoot;
        this.candidates = candidates;
        this.bestBN = candidates.get(0);
    }

    @Override
    public Object getModel() {
        return treeRoot;
    }

    @Override
    public void setModel(Object model) {
        if (!(model instanceof TreeNode)) {
            throw new IllegalArgumentException("The model must be object of the TreeNode class");
        }

        this.treeRoot = (TreeNode) model;
    }

    @Override
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        Dag dag = this.bestBN.getModel();

        double bdeu = getScore(data);
        if (this.lastBN != null) {
            double lastBDeu = this.lastBN.getScore();
            if (bdeu <= lastBDeu) {
                return;
            }
        }

        int smhd = ExperimentUtils.calculateSMHD(data, dag);
        int threads = Runtime.getRuntime().availableProcessors();

        String completePath = path + "results/" + epoch + "/" + data.getName() + "_" + operation + "_" + nClients + "_" + id + ".csv";
        String header = "bbdd,algorithm,maxEdges,exploitation,probSwap,nSwaps,nClients,id,iteration,instances,threads,bdeu,SMHD,edges,time(s)\n";
        String results = data.getName() + "," +
                operation + "," +
                nClients + "," +
                id + "," +
                iteration + "," +
                data.getNInstances() + "," +
                threads + "," +
                bdeu + "," +
                smhd + "," +
                dag.getEdges().size() + "," +
                time + "\n";

        System.out.println(results);

        ExperimentUtils.saveExperiment(completePath, header, results);
    }

    @Override
    public double getScore() {
        return bestBN.getScore();
    }

    @Override
    public double getScore(Data data) {
        return bestBN.getScore(data);
    }

    /**
     * Calculate the best BN of the candidates.
      * @param data The data to be used to calculate the metrics.
     */
    public void calculateBestBN(Data data) {
        double score = bestBN.getScore(data);
        for (BN candidate : candidates) {
            double candidateScore = candidate.getScore(data);
            if (candidateScore > score) {
                score = candidateScore;
                bestBN = candidate;
            }
        }
    }

    @Override
    public String toString() {
        return treeRoot.toString();
    }

    public BN getBestBN() {
        return bestBN;
    }
}
