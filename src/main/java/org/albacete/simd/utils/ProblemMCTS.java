package org.albacete.simd.utils;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.EdgeListGraph_n;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.BDeuScore;
import org.albacete.simd.threads.GESThread;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class ProblemMCTS extends Problem {

    /**
     * Caches scores for discrete search.
     */
    private final ConcurrentHashMap<String,Double> localScoreCache = new ConcurrentHashMap<>();

    public static double emptyGraphScore;

    public static int nInstances;

    public ProblemMCTS (DataSet dataSet){
        super(dataSet);

        Graph graph = new EdgeListGraph_n(new LinkedList<Node>(getVariables()));
        emptyGraphScore = GESThread.scoreGraph(graph, this);
        nInstances = dataSet.getNumRows();
    }


    public ProblemMCTS(String file){
        super(file);
    }

    public ConcurrentHashMap<String,Double> getConcurrentHashMap() {
        return localScoreCache;
    }

    public Node getNode(String name){
        for (Node node: getVariables()) {
            if(node.getName().equals(name))
                return node;
        }
        return null;
    }
    
    public Node getNode(int id){
        for (Node node: getVariables()) {
            if(hashIndices.get(node) == id)
                return node;
        }
        return null;
    }


}
