package org.albacete.simd.threads;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Problem;

import java.util.*;

import static org.albacete.simd.utils.Utils.pdagToDag;

public class BESThread extends GESThread {

    private static int threadCounter = 1;

    /**
     * Constructor of ThFES with an initial DAG
     *
     * @param problem    object containing information of the problem such as data or variables.
     * @param initialDag initial DAG with which the BES stage starts with.
     * @param subset     subset of edges the fes stage will try to remove 
     */
    public BESThread(Problem problem, Graph initialDag, Set<Edge> subset) {

        this.problem = problem;
        setInitialGraph(initialDag);
        setSubSetSearch(subset);

        // Setting structure prior and sample prior
        setStructurePrior(0.001);
        setSamplePrior(10.0);
        this.id = threadCounter;
        threadCounter++;
        this.isForwards = false;
    }

    /**
    Run method from {@link Runnable Runnable} interface. The method executes the {@link #search()} search} method to remove
    edges from the initial graph.
     */
    @Override
    public void run() {
        this.currentGraph = search();
        pdagToDag(this.currentGraph);
    }

    /**
     * Search method that explores the data and currentGraph to return a better Graph
     * @return PDAG that contains either the result of the BES or FES method.
     */
    private Graph search() {
        startTime = System.currentTimeMillis();
        numTotalCalls=0;
        numNonCachedCalls=0;

        Graph graph = new EdgeListGraph(this.initialDag);

        // Method 1-- original.
        double scoreInitial = scoreDag(graph);

        // Do backward search.
        bes(graph, scoreInitial);

        long endTime = System.currentTimeMillis();
        this.elapsedTime = endTime - startTime;

        double newScore = scoreDag(graph);
        // If we improve the score, return the new graph
        if (newScore > scoreInitial) {
            this.modelBDeu = newScore;
            this.flag = true;
            return graph;
        } else {
            this.modelBDeu = scoreInitial;
            this.flag = false;
            return this.initialDag;
        }
    }

    /**
     * Backward equivalence search.
     *
     * @param graph The graph in the state prior to the backward equivalence
     *              search.
     * @param score The score in the state prior to the backward equivalence
     *              search
     * @return the score in the state after the BES method.
     *         Note that the graph is changed as a side-effect to its state after
     *         the backward equivalence search.
     */
    private double bes(Graph graph, double score) {
        //System.out.println("** BACKWARD EQUIVALENCE SEARCH");
        double bestScore = score;
        double bestDelete;

        x_d = null;
        y_d = null;
        h_0 = null;

        //System.out.println("Initial Score = " + nf.format(bestScore));
        // Calling fs to calculate best edge to add.
        bestDelete = bs(graph,bestScore);

        while(x_d != null){
            // Deleting edge
            delete(x_d,y_d,h_0, graph);

            //PDAGtoCPDAG
            rebuildPattern(graph);

            bestScore = bestDelete;

            // Checking that the maximum number of edges has not been reached
            if (getMaxNumEdges() != -1 && graph.getNumEdges() > getMaxNumEdges()) {
                break;
            }

            // Executing BS function to calculate the best edge to be deleted
            bestDelete = bs(graph,bestScore);

            // Indicating that the thread has deleted an edge to the graph
            this.flag = true;

        }
        return bestScore;
    }

    /**
     * BS method of the BES algorithm. It finds the best possible edge, alongside with the subset h_0 that is best suited
     * for deletion in the current graph.
     * @param graph current graph of the thread.
     * @param initialScore score the current graph has.
     * @return score of the best possible deletion found.
     */
    private double bs(Graph graph, double initialScore){
        x_d = y_d = null;
        h_0 = null;
        
        Set<Edge> edgesInGraph = graph.getEdges();
        
        if (!edgesInGraph.isEmpty()) {
            EdgeSearch[] arrScores = new EdgeSearch[edgesInGraph.size()];
            List<Edge> edges = new ArrayList<>(edgesInGraph);

            Arrays.parallelSetAll(arrScores, e-> scoreEdge(graph, edges.get(e), initialScore));

            List<EdgeSearch> list = Arrays.asList(arrScores);
            EdgeSearch max = Collections.max(list);

            if (max.score > initialScore) {
                x_d = max.edge.getNode1();
                y_d = max.edge.getNode2();
                h_0 = max.hSubset;
            }
            
            return max.score;
        }
        return initialScore;
    }
    
    private EdgeSearch scoreEdge(Graph graph, Edge edge, double initialScore) {
        Node _x = edge.getNode1();
        Node _y = edge.getNode2();

        List<Node> hNeighbors = getSubsetOfNeighbors(_x, _y, graph);
        List<HashSet<Node>> hSubsets = generatePowerSet(hNeighbors);

        double changueEval;
        double evalScore;
        double bestScore = initialScore;
        HashSet<Node> bestSubSet = new HashSet<>();

        for (HashSet<Node> hSubset : hSubsets) {
            changueEval = deleteEval(_x, _y, hSubset, graph);

            evalScore = initialScore + changueEval;

            if (evalScore > bestScore) {
                // START TEST 1
                HashSet<Node> naYXH = findNaYX(_x, _y, graph);
                naYXH.removeAll(hSubset);
                if (isClique(naYXH, graph)) {
                    // END TEST 1
                    bestScore = evalScore;
                    bestSubSet = hSubset;
                }
            }
        }
        return new EdgeSearch(bestScore, bestSubSet, edge);
    }

    public static List<HashSet<Node>> generatePowerSet2(List<Node> nodes) {
        List<HashSet<Node>> subsets = new ArrayList<>();

        for (int i = 0; i < Math.pow(2, nodes.size()); i++) {
            HashSet<Node> newSubSet = new HashSet<>();
            String selection = Integer.toBinaryString(i);
            for (int j = selection.length() - 1; j >= 0; j--) {
                if (selection.charAt(j) == '1') {
                    newSubSet.add(nodes.get(selection.length() - j - 1));
                }
            }
            subsets.add(newSubSet);
        }

        return subsets;
    }

    public static List<HashSet<Node>> generatePowerSet(List<Node> nodes) {
        List<HashSet<Node>> subsets = new ArrayList<>();
        int n = nodes.size();

        for (int i = 0; i < (1 << n); i++) {  // Usa (1 << n) en lugar de Math.pow(2, n) para mayor eficiencia.
            HashSet<Node> newSubSet = new HashSet<>();
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) != 0) {  // Chequea si el bit j de i está activado.
                    newSubSet.add(nodes.get(j));
                }
            }
            subsets.add(newSubSet);
        }

        return subsets;
    }

}
