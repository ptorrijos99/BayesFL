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
 *    mAnDETree_mAnDE.java
 *    Copyright (C) 2024 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.algorithms;

import bayesfl.data.CPT_Instances;
import bayesfl.data.Data;
import bayesfl.model.Model;
import bayesfl.model.mAnDETree;
import org.albacete.simd.mAnDE.mAnDE;
import weka.core.Instances;

public class mAnDETree_mAnDE implements LocalAlgorithm {

    private int n = 1;
    private int nTrees = 100;

    private double bagSize = 100;
    private String ensemble = "RandomForest";
    private double addNB = 0;

    public mAnDETree_mAnDE() {}

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

    @Override
    public Model buildLocalModel(Data data) {
        return null;
    }

    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        if (!(data instanceof CPT_Instances)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        Instances instances = (Instances) data.getData();

        mAnDE algorithm = getAlgorithm();

        try {
            algorithm.checkData(instances);
            algorithm.buildTrees();
        } catch (Exception e) { e.printStackTrace(); }

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

    public mAnDE getAlgorithm() {
        mAnDE algorithm = new mAnDE();
        algorithm.setN(this.n);
        algorithm.setnTrees(this.nTrees);
        algorithm.setBagSize(this.bagSize);
        algorithm.setEnsemble(this.ensemble);
        algorithm.setAddNB(this.addNB);
        return algorithm;
    }
}
