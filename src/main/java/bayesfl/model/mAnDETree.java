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
 *    mAnDETree.java
 *    Copyright (C) 2024 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.model;

import bayesfl.algorithms.mAnDETree_mAnDE;
import bayesfl.data.CPT_Instances;
import bayesfl.data.Data;
import bayesfl.experiments.ExperimentUtils;
import org.albacete.simd.mAnDE.mAnDE;
import org.albacete.simd.mAnDE.mSPnDE;
import weka.core.Instances;

import java.util.concurrent.ConcurrentHashMap;

import static bayesfl.experiments.ExperimentUtils.getClassificationMetrics;

public class mAnDETree implements Model {

    private ConcurrentHashMap<Object, mSPnDE> models;

    private final mAnDETree_mAnDE algorithm;

    private double score;

    /**
     * Constructor that adds the algorithm, so we can use them to calculate the score
     */
    public mAnDETree(ConcurrentHashMap<Object, mSPnDE> models, mAnDETree_mAnDE algorithm) {
        this.models = models;
        this.algorithm = algorithm;
    }

    @Override
    public ConcurrentHashMap<Object, mSPnDE> getModel() {
        return models;
    }

    @Override
    public void setModel(Object model) {
        if (!(model instanceof ConcurrentHashMap)) {
            throw new IllegalArgumentException("Model must be a ConcurrentHashMap<Object, mSPnDE>");
        }

        this.models = (ConcurrentHashMap<Object, mSPnDE>) model;
    }

    @Override
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        if (!(data instanceof CPT_Instances)) {
            throw new IllegalArgumentException("The data must be object of the CPT_Instances class");
        }

        mAnDE mAnDE = algorithm.getAlgorithm();

        int threads = Runtime.getRuntime().availableProcessors();

        Instances train = (Instances) data.getData();
        Instances test = ((CPT_Instances) data).getTest();

        double start = System.currentTimeMillis();
        try {
            mAnDE.checkData(train);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        System.out.println("SaveStats: " + this.models);

        mAnDE.mSPnDEs = this.models;
        mAnDE.calculateTables_mSPnDEs();
        double timeTables = (System.currentTimeMillis() - start) / 1000;

        start = System.currentTimeMillis();
        double[] trainMetrics = getClassificationMetrics(mAnDE, train);
        double timeTrain = (System.currentTimeMillis() - start) / 1000;

        start = System.currentTimeMillis();
        double[] testMetrics = getClassificationMetrics(mAnDE, test);
        double timeTest = (System.currentTimeMillis() - start) / 1000;

        String completePath = path + "results/" + epoch + "/" + data.getName() + "_" + operation + "_" + nClients + "_" + id + ".csv";
        String header = "bbdd,id,cv,algorithm,seed,nTrees,bagSize,ensemble,addNB,nClients,iteration,instances,threads,trAcc,trPr,trRc,trF1,teAcc,teRp,teRc,teF1,time,timeTables,timeTrain,timeTest\n";
        String results = data.getName() + "," +
                operation + "," +
                nClients + "," +
                iteration + "," +
                data.getNInstances() + "," +
                threads + "," +
                trainMetrics[0] + "," +
                trainMetrics[1] + "," +
                trainMetrics[2] + "," +
                trainMetrics[3] + "," +
                testMetrics[0] + "," +
                testMetrics[1] + "," +
                testMetrics[2] + "," +
                testMetrics[3] + "," +
                time + "," +
                timeTables + "," +
                timeTrain + "," +
                timeTest + "\n";

        System.out.println(results);

        ExperimentUtils.saveExperiment(completePath, header, results);
    }

    @Override
    public double getScore() {
        return this.score;
    }

    @Override
    public double getScore(Data data) {
        mAnDE mAnDE = algorithm.getAlgorithm();

        Instances train = (Instances) data.getData();

        try {
            mAnDE.checkData(train);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        mAnDE.mSPnDEs = this.models;
        mAnDE.calculateTables_mSPnDEs();

        return getClassificationMetrics(mAnDE, train)[0];
    }

    public mAnDETree_mAnDE getAlgorithm() {
        return algorithm;
    }
}
