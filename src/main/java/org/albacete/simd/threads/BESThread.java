package org.albacete.simd.threads;

import consensusBN.PowerSet;
import consensusBN.PowerSetFabric;
import consensusBN.SubSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Problem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import static org.albacete.simd.utils.Utils.pdagToDag;

@SuppressWarnings("DuplicatedCode")
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
        if (!S.isEmpty()) {
            startTime = System.currentTimeMillis();
            numTotalCalls=0;
            numNonCachedCalls=0;
            //localScoreCache.clear();

            Graph graph = new EdgeListGraph_n(this.initialDag);
            //buildIndexing(graph);

            // Method 1-- original.
            double scoreInitial = scoreDag(graph);

            // Do backward search.
            bes(graph, scoreInitial);

            long endTime = System.currentTimeMillis();
            this.elapsedTime = endTime - startTime;

            double newScore = scoreDag(graph);
            System.out.println(" ["+getId()+"] BES New Score: " + newScore + ", Initial Score: " + scoreInitial);
            // If we improve the score, return the new graph
            if (newScore > scoreInitial) {
                this.modelBDeu = newScore;
                this.flag = true;
                return graph;
            } else {
                //System.out.println("   ["+getId()+"] ELSE");
                this.modelBDeu = scoreInitial;
                this.flag = false;
                return this.initialDag;
            }
        } else return this.initialDag;
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
            // Changing best score because x_d, and y_d are not null
            bestScore = bestDelete;

            // Deleting edge
            //System.out.println("Thread " + getId() + " deleting: (" + x_d + ", " + y_d + ", " + h_0+ ")");
            delete(x_d,y_d,h_0, graph);
            
            // Checking cycles?
            //System.out.println("  Ciclos: " + graph.existsDirectedCycle());

            //PDAGtoCPDAG
            rebuildPattern(graph);
            
            // Printing score
            /*if (!h_0.isEmpty())
                System.out.println("Score: " + nf.format(bestScore) + " (+" + nf.format(bestDelete-score) +")\tOperator: " + graph.getEdge(x_d, y_d) + " " + h_0);
            else
                System.out.println("Score: " + nf.format(bestScore) + " (+" + nf.format(bestDelete-score) +")\tOperator: " + graph.getEdge(x_d, y_d));
            */
            bestScore = bestDelete;
            //System.out.println("    Real Score" + scoreGraph(graph, problem));

            // Checking that the maximum number of edges has not been reached
            if (getMaxNumEdges() != -1 && graph.getNumEdges() > getMaxNumEdges()) {
                //System.out.println("Maximum edges reached");
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
        //   	System.out.println("\n** BACKWARD ELIMINATION SEARCH");
        //   	System.out.println("Initial Score = " + nf.format(initialScore));

        PowerSetFabric.setMode(PowerSetFabric.MODE_BES);

        x_d = y_d = null;
        h_0 = null;
        
        Set<Edge> edgesInGraph = graph.getEdges();
        
        if (!edgesInGraph.isEmpty()) {
            EdgeSearch[] arrScores = new EdgeSearch[edgesInGraph.size()];
            List<Edge> edges = new ArrayList<>(edgesInGraph);

            Arrays.parallelSetAll(arrScores, e->{
                return scoreEdge(graph, edges.get(e), initialScore);
            });

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
        // Checking if the edge is actually inside the graph
        if(S.contains(edge)) {
            Node _x = Edges.getDirectedEdgeTail(edge);
            Node _y = Edges.getDirectedEdgeHead(edge);

            List<Node> hNeighbors = getSubsetOfNeighbors(_x, _y, graph);
            PowerSet hSubsets = PowerSetFabric.getPowerSet(_x, _y, hNeighbors);
            
            double changueEval;
            double evalScore;
            double bestScore = initialScore; 
            SubSet bestSubSet = new SubSet();
            
            while(hSubsets.hasMoreElements()) {
                SubSet hSubset = hSubsets.nextElement();
                changueEval = deleteEval(_x, _y, hSubset, graph);
                
                evalScore = initialScore + changueEval;

                if (evalScore > bestScore) {
                    // START TEST 1
                    List<Node> naYXH = findNaYX(_x, _y, graph);
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
        return new EdgeSearch(initialScore, new SubSet(), edge);
    }
    
    /**
     * BS method of the BES algorithm. It finds the best possible edge, alongside with the subset h_0 that is best suited
     * for deletion in the current graph.
     * @param graph current graph of the thread.
     * @param initialScore score the current graph has.
     * @return score of the best possible deletion found.
     */
    private double bs2(Graph graph, double initialScore){
        //   	System.out.println("\n** BACKWARD ELIMINATION SEARCH");
        //   	System.out.println("Initial Score = " + nf.format(initialScore));

        PowerSetFabric.setMode(PowerSetFabric.MODE_BES);
        double bestScore = initialScore;

        x_d = y_d = null;
        h_0 = null;

        List<Edge> edges = new ArrayList<>(S);

/*
        for (TupleNode tupleNode : this.S) {
            Node _x = tupleNode.x;
            Node _y = tupleNode.y;
            // Adding Edges to check
            edges.add(Edges.directedEdge(_x, _y));
            edges.add(Edges.directedEdge(_y, _x));
        }
*/
        for (Edge edge : edges) {

            //Checking time
            /*if(isTimeout()) {
                System.out.println("Timeout in BESTHREAD id: " + getId());
                break;
            }*/

            // Checking if the edge is actually inside the graph
            if(!graph.containsEdge(edge))
                continue;

            Node _x = Edges.getDirectedEdgeTail(edge);
            Node _y = Edges.getDirectedEdgeHead(edge);

            List<Node> hNeighbors = getSubsetOfNeighbors(_x, _y, graph);
            //                List<Set<Node>> hSubsets = powerSet(hNeighbors);
            PowerSet hSubsets= PowerSetFabric.getPowerSet(_x,_y,hNeighbors);

            while(hSubsets.hasMoreElements()) {
                SubSet hSubset=hSubsets.nextElement();
                double deleteEval = deleteEval(_x, _y, hSubset, graph);
                double evalScore = initialScore + deleteEval;

                //                    System.out.println("Attempt removing " + _x + "-->" + _y + "(" +
                //                            evalScore + ")");

                if (!(evalScore > bestScore)) {
                    continue;
                }

                // START TEST 1
                List<Node> naYXH = findNaYX(_x, _y, graph);
                naYXH.removeAll(hSubset);
                if (!isClique(naYXH, graph)) {
                    continue;
                }
                // END TEST 1

                bestScore = evalScore;
                x_d = _x;
                y_d = _y;
                h_0 = hSubset;
            }

        }

        return bestScore;
    }
    
}
