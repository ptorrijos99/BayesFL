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
import edu.cmu.tetrad.graph.*;

import org.albacete.simd.utils.Utils;
import bayesfl.data.Data;

import java.util.*;

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

    /**
     * Anonymizes variable names in the DAG by replacing them with numerical identifiers
     * based on lexicographical ordering. This prevents exposing sensitive variable names
     * when exchanging graph structures between participants.
     */
    public void anonymizeVariables() {
        Dag originalDag = this.dag;

        // Collect all variable names and sort lexicographically
        List<String> variableNames = new ArrayList<>();
        for (Node node : originalDag.getNodes()) {
            variableNames.add(node.getName());
        }
        Collections.sort(variableNames);

        // Create mapping from original name to anonymized identifier
        Map<String, String> nameToAnonMap = new HashMap<>();
        for (int i = 0; i < variableNames.size(); i++) {
            nameToAnonMap.put(variableNames.get(i), String.valueOf(i));
        }

        // Create new anonymized DAG
        Dag anonDag = new Dag();

        // Add anonymized nodes
        for (String originalName : variableNames) {
            Node anonNode = new GraphNode(nameToAnonMap.get(originalName));
            anonDag.addNode(anonNode);
        }

        // Rebuild edges with anonymized nodes
        for (Edge edge : originalDag.getEdges()) {
            Node origNode1 = edge.getNode1();
            Node origNode2 = edge.getNode2();

            Node anonNode1 = anonDag.getNode(nameToAnonMap.get(origNode1.getName()));
            Node anonNode2 = anonDag.getNode(nameToAnonMap.get(origNode2.getName()));

            Edge anonEdge = new Edge(anonNode1, anonNode2, edge.getEndpoint1(), edge.getEndpoint2());
            anonDag.addEdge(anonEdge);
        }

        // Update the DAG in this BN instance
        this.dag = anonDag;
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
        int shd = ExperimentUtils.calculateSHD(data, this.dag);
        int fusSim = ExperimentUtils.calculateFusSim(data, this.dag);
        int threads = Runtime.getRuntime().availableProcessors();
        int tw = Utils.getTreeWidth(this.dag);

        String completePath = path + "results/" + epoch + "/" + data.getName() + "_" + operation + "_" + nClients + "_" + id + ".csv";
        String header = "bbdd,algorithm,maxEdges,fusionC,limitC,convergence,fusionS,limitS,nClients,id,iteration,instances,threads,bdeu,SMHD,SHD,fusSim,edges,tw,time(s)\n";
        String results = data.getName() + "," +
                        operation + "," +
                        nClients + "," +
                        id + "," +
                        iteration + "," +
                        data.getNInstances() + "," +
                        threads + "," +
                        this.score + "," +
                        smhd + "," +
                        shd + "," +
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


