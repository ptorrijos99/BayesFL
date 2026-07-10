/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2026 Universidad de Castilla-La Mancha, España
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
 *    mAnDETree_mAnDE.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.algorithms;

import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.model.Model;
import bayesfl.model.mAnDETree;
import org.albacete.simd.mAnDE.mAnDE;
import weka.core.Instances;
import weka.filters.Filter;
import bayesfl.algorithms.Dummy;

public class mAnDETree_mAnDE implements LocalAlgorithm {

    private int n = 1;
    private int nTrees = 100;

    private double bagSize = 100;
    private String ensemble = "RandomForest";
    private double addNB = 0;
    private double[] addNBValues = null;
    private double[][] cutPoints = null;

    /** Conditional-Gaussian (HAODE-style) parameterisation for continuous data. */
    private boolean continuous = false;
    /** Empirical-Bayes prior dof for the CG variance shrinkage (only if continuous). */
    private double cgPriorVarDof = 3.0;
    /** Build the blended NB on the DISCRETIZED data (categorical NB) in continuous mode. */
    private boolean cgNBDiscretized = false;

    public mAnDETree_mAnDE() {
    }

    public mAnDETree_mAnDE(int n) {
        this.n = n;
    }

    public mAnDETree_mAnDE(int n, int nTrees, double bagSize, String ensemble, double addNB) {
        this(n);
        this.nTrees = nTrees;
        this.bagSize = bagSize;
        this.ensemble = ensemble;
        this.addNB = addNB;
    }

    public mAnDETree_mAnDE(int n, int nTrees, double bagSize, String ensemble, double addNB,
                           boolean continuous, double cgPriorVarDof) {
        this(n, nTrees, bagSize, ensemble, addNB);
        this.continuous = continuous;
        this.cgPriorVarDof = cgPriorVarDof;
    }

    public mAnDETree_mAnDE(int n, int nTrees, double bagSize, String ensemble, double addNB, double[][] cutPoints) {
        this(n);
        this.nTrees = nTrees;
        this.bagSize = bagSize;
        this.ensemble = ensemble;
        this.addNB = addNB;
        this.cutPoints = cutPoints;
    }

    @Override
    public Model buildLocalModel(Data data) {
        return null;
    }

    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        if (!(data instanceof Weka_Instances)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        Instances instances = (Instances) data.getData();

        if (this.cutPoints != null) {
            Dummy filter = new Dummy();
            filter.setCutPoints(this.cutPoints);
            try {
                filter.setInputFormat(instances);
                instances = Filter.useFilter(instances, filter);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        mAnDE algorithm = getAlgorithm();

        try {
            algorithm.checkData(instances);
            algorithm.buildTrees();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new mAnDETree(algorithm.mSPnDEs, this);
    }

    @Override
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return null;
    }

    @Override
    public String getAlgorithmName() {
        return "mA" + n + "DE";
    }

    @Override
    public String getRefinementName() {
        return "None";
    }

    public void setAddNBValues(double[] values) { this.addNBValues = values; }
    public double[] getAddNBValues() { return addNBValues; }
    public double getAddNB() { return addNB; }

    public void setContinuous(boolean continuous) { this.continuous = continuous; }
    public boolean isContinuous() { return continuous; }
    public void setCgPriorVarDof(double d0) { this.cgPriorVarDof = d0; }
    public double getCgPriorVarDof() { return cgPriorVarDof; }
    public void setCgNBDiscretized(boolean v) { this.cgNBDiscretized = v; }
    public boolean isCgNBDiscretized() { return cgNBDiscretized; }

    public mAnDE getAlgorithm() {
        mAnDE algorithm = new mAnDE();
        algorithm.setN(this.n);
        algorithm.setnTrees(this.nTrees);
        algorithm.setBagSize(this.bagSize);
        algorithm.setEnsemble(this.ensemble);
        algorithm.setAddNB(this.addNB);
        algorithm.setContinuous(this.continuous);
        algorithm.setCgPriorVarDof(this.cgPriorVarDof);
        algorithm.setCgNBDiscretized(this.cgNBDiscretized);
        return algorithm;
    }
}
