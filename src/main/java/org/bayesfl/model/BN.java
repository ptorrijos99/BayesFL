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
 *    BN.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.bayesfl.model;

import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import org.albacete.simd.utils.Utils;
import org.bayesfl.data.Data;
import org.bayesfl.data.BN_DataSet;
import org.w3c.dom.Document;
import weka.classifiers.bayes.net.BIFReader;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;

public class BN implements Model {

    private Dag_n dag;

    public BN() {
        this.dag = new Dag_n();
    }

    public BN(Dag_n dag) {
        this.dag = dag;
    }

    public BN(Graph graph) {
        this.dag = new Dag_n(Utils.removeInconsistencies(graph));
    }

    @Override
    public Dag_n getModel() {
        return dag;
    }

    @Override
    public void setModel(Object model) {
        if (!(model instanceof Dag_n)) {
            throw new IllegalArgumentException("The model must be object of the BN class");
        }

        this.dag = (Dag_n) model;
    }

    @Override
    public void printStats() {
        System.out.println("| Total Nodes of DAG: " + this.dag.getNodes().size());
        System.out.println("| Total Edges of DAG: " + this.dag.getEdges().size());
    }

    @Override
    public void printStats(Data data) {
        printStats();

        if (!(data instanceof BN_DataSet)) {
            throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
        }

        // Calculate bdeu score
        BDeuScore bdeu = new BDeuScore((DataSet) data.getData());
        Fges fges = new Fges(bdeu);
        System.out.println("| BDeu score: " + fges.scoreDag(this.dag));

        if (((BN_DataSet) data).getOriginalBNPath() != null) {
            printOriginalBNStats(((BN_DataSet) data).getOriginalBNPath());
        }
    }
    
    public void printOriginalBNStats(String patch) {
        try {
            MlBayesIm originalBN = readOriginalBayesianNetwork(patch);
            System.out.println("| SMHD score: " + Utils.SHD(Utils.removeInconsistencies(originalBN.getDag()), this.dag));
        } catch (Exception e) { e.printStackTrace(); }
    }

    /**
     * Read the original Bayesian Network from the BIF file in the netPath.
     * @return The original Bayesian Network.
     * @throws Exception If the file is not found.
     */
    private MlBayesIm readOriginalBayesianNetwork(String netPath) throws Exception {
        final PrintStream err = new PrintStream(System.err);
        System.setErr(new PrintStream(OutputStream.nullOutputStream()));

        BIFReader bayesianReader = new BIFReader();
        bayesianReader.processFile(netPath);

        System.setErr(err);

        // Transforming the BayesNet into a BayesPm
        BayesPm bayesPm = Utils.transformBayesNetToBayesPm(bayesianReader);

        return new MlBayesIm(bayesPm);
    }

    @Override
    public String toString() {
        return dag.toString();
    }
}


