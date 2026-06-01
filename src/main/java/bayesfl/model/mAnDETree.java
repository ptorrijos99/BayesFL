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
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.experiments.utils.ExperimentUtils;
import org.albacete.simd.mAnDE.mAnDE;
import org.albacete.simd.mAnDE.mSP1DE;
import org.albacete.simd.mAnDE.mSP2DE;
import org.albacete.simd.mAnDE.mSPnDE;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import static bayesfl.experiments.utils.ExperimentUtils.getClassificationMetrics;

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
        if (!(data instanceof Weka_Instances)) {
            throw new IllegalArgumentException("The data must be object of the CPT_Instances class");
        }

        mAnDE mAnDE = algorithm.getAlgorithm();

        int threads = Runtime.getRuntime().availableProcessors();

        Instances train = ((Weka_Instances) data).getTrain();
        Instances test = ((Weka_Instances) data).getTest();

        double start = System.currentTimeMillis();
        try {
            mAnDE.checkData(train);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        mAnDE.mSPnDEs = this.models;

        // Always build NB so any γ > 0 can use it
        double[] gammas = algorithm.getAddNBValues();
        if (gammas != null) {
            mAnDE.setAddNB(1.0);
        }
        mAnDE.calculateTables_mSPnDEs();
        double timeTables = (System.currentTimeMillis() - start) / 1000;

        // SPODE stats (γ-independent)
        double var = 0;
        double max = 0;
        double min = Double.POSITIVE_INFINITY;
        for (mSPnDE a : mAnDE.mSPnDEs.values()) {
            if (a.getNChildren() > max)
                max = a.getNChildren();
            if (a.getNChildren() < min)
                min = a.getNChildren();
            var += a.getNChildren();
        }

        String nameExceptIdCv = data.getName();
        int commaIdx = nameExceptIdCv.indexOf(",");
        nameExceptIdCv = commaIdx >= 0 ? nameExceptIdCv.substring(0, commaIdx) : nameExceptIdCv;

        if (gammas == null) {
            gammas = new double[]{algorithm.getAddNB()};
        }

        String header = "bbdd,id,cv,node,algorithm,seed,nTrees,bagSize,ensemble,addNB,nClients,iteration,instances,threads,spodes,varPerSpode,maxSpode,minSpode,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time,timeTables\n";

        for (double gamma : gammas) {
            mAnDE.setAddNB(gamma);

            String trainMetrics = getClassificationMetrics(mAnDE, train);
            String testMetrics = getClassificationMetrics(mAnDE, test);

            // Replace addNB (last field) in operation string with this γ
            int lastComma = operation.lastIndexOf(',');
            String gammaOp = operation.substring(0, lastComma + 1) + gamma;

            String completePath = path + "results/" + epoch + "/" + nameExceptIdCv + "_" + gammaOp + "_" + nClients + ".csv";
            String results = data.getName() + "," +
                    gammaOp + "," +
                    nClients + "," +
                    iteration + "," +
                    data.getNInstances() + "," +
                    threads + "," +
                    mAnDE.mSPnDEs.size() + "," +
                    (var / mAnDE.mSPnDEs.size()) + "," +
                    max + "," +
                    min + "," +
                    trainMetrics +
                    testMetrics +
                    time + "," +
                    timeTables + "\n";

            System.out.println(results);
            ExperimentUtils.saveExperiment(completePath, header, results);
        }
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

        Evaluation evaluation = null;
        try {
            evaluation = new Evaluation(train);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        String num = getClassificationMetrics(mAnDE, train).split(",")[0];
        return Double.parseDouble(num);
    }

    public mAnDETree_mAnDE getAlgorithm() {
        return algorithm;
    }

    /**
     * Converts the mSPnDE structure map to a list of super-parent index arrays
     * compatible with PT_AnDE. Each entry corresponds to one SPODE:
     * - For mSP1DE: int[]{xi} (single super-parent)
     * - For mSP2DE: int[]{xi1, xi2} (pair of super-parents, canonically ordered)
     *
     * @return List of super-parent index arrays for use with PT_AnDE.
     */
    public List<int[]> toCombinations() {
        List<int[]> combinations = new ArrayList<>();

        for (mSPnDE spode : models.values()) {
            if (spode instanceof mSP1DE sp1) {
                // For mSP1DE, the key in the map IS the xi index (Integer)
                // We need to extract it from the map
                for (Object key : models.keySet()) {
                    if (models.get(key) == spode) {
                        combinations.add(new int[]{((Integer) key)});
                        break;
                    }
                }
            } else if (spode instanceof mSP2DE sp2) {
                // For mSP2DE, the key is a long encoding: (xi1 << 32) | xi2
                for (Object key : models.keySet()) {
                    if (models.get(key) == spode) {
                        long k = (Long) key;
                        int xi1 = (int) (k >> 32);
                        int xi2 = (int) k;
                        // Ensure canonical order (xi1 < xi2)
                        if (xi1 > xi2) { int tmp = xi1; xi1 = xi2; xi2 = tmp; }
                        combinations.add(new int[]{xi1, xi2});
                        break;
                    }
                }
            }
        }

        return combinations;
    }
}
