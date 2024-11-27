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
 *    BN.java
 *    Copyright (C) 2024 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.model;

import bayesfl.experiments.utils.ExperimentUtils;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;

import org.albacete.simd.utils.Utils;
import bayesfl.data.Data;

public class BN implements Model {
    
    private Dag dag;
    
    private double score;

    public BN() {
        this.dag = new Dag();
    }

    public BN(Dag dag) {
        this.dag = dag;
    }

    public BN(Graph graph) {
        this.dag = new Dag(Utils.removeInconsistencies(graph));
    }

    @Override
    public Dag getModel() {
        return dag;
    }

    @Override
    public void setModel(Object model) {
        if (!(model instanceof Dag)) {
            throw new IllegalArgumentException("The model must be object of the BN class");
        }

        this.dag = (Dag) model;
    }

    @Override
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        this.score = ExperimentUtils.calculateBDeu(data, this.dag);
        int smhd = ExperimentUtils.calculateSMHD(data, this.dag);
        int fusSim = ExperimentUtils.calculateFusSim(data, this.dag);
        int threads = Runtime.getRuntime().availableProcessors();
        int tw = Utils.getTreeWidth(this.dag);

        String completePath = path + "results/" + epoch + "/" + data.getName() + "_" + operation + "_" + nClients + "_" + id + ".csv";
        String header = "bbdd,algorithm,maxEdges,fusionC,limitC,refinement,fusionS,limitS,nClients,id,iteration,instances,threads,bdeu,SMHD,fusSim,edges,tw,time(s)\n";
        String results = data.getName() + "," +
                        operation + "," +
                        nClients + "," +
                        id + "," +
                        iteration + "," +
                        data.getNInstances() + "," +
                        threads + "," +
                        this.score + "," +
                        smhd + "," +
                        fusSim + "," +
                        this.dag.getEdges().size() + "," +
                        tw + "," +
                        time + "\n";
        
        System.out.println(results);

        ExperimentUtils.saveExperiment(completePath, header, results);
    }
    
    @Override
    public double getScore(){
        return this.score;
    }
    
    @Override
    public double getScore(Data data) {
        this.score = ExperimentUtils.calculateBDeu(data, this.dag);
        return this.score;
    }
    
    @Override
    public String toString() {
        return dag.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof BN bn)) return false;

        if (this.dag.getNodes().size() != bn.dag.getNodes().size()) return false;

        return this.dag.getEdges().equals(bn.dag.getEdges());
    }

    @Override
    public int hashCode() {
        return this.dag.getEdges().hashCode();
    }
}


