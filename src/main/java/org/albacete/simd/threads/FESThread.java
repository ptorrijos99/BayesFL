package org.albacete.simd.threads;

import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.utils.MeekRules;
import edu.cmu.tetrad.search.utils.GraphSearchUtils;
import org.albacete.simd.utils.Problem;

import java.util.*;

import java.util.stream.Collectors;

import static org.albacete.simd.utils.Utils.pdagToDag;

@SuppressWarnings("DuplicatedCode")
public class FESThread extends GESThread {

    private static int threadCounter = 1;
    
    private final boolean speedUp;
    private final boolean update;
    private final boolean parallel;

    /**
     * Constructor of FESThread with an initial DAG
     *
     * @param problem object containing all the information of the problem
     * @param initialDag initial DAG with which the FES stage starts with, if
     * it's null, use the other constructor
     * @param subset subset of edges the fes stage will try to add to the
     * resulting graph
     * @param maxIt maximum number of iterations allowed in the fes stage
     * @param speedUp
     */
    public FESThread(Problem problem, Graph initialDag, Set<Edge> subset, int maxIt, boolean speedUp, boolean update, boolean parallel) {
        this(problem, subset, maxIt, speedUp, update, parallel);
        this.initialDag = initialDag;
    }

    /**
     * Constructor of FESThread with an initial DataSet
     *
     * @param problem object containing information of the problem such as data
     * or variables.
     * @param subset subset of edges the fes stage will try to add to the
     * resulting graph
     * @param maxIt maximum number of iterations allowed in the fes stage
     * @param speedUp
     */
    public FESThread(Problem problem, Set<Edge> subset, int maxIt, boolean speedUp, boolean update, boolean parallel) {
        this.problem = problem;
        this.initialDag = new EdgeListGraph(new LinkedList<>(getVariables()));
        setSubSetSearch(subset);
        setMaxIt(maxIt);
        this.id = threadCounter;
        threadCounter++;
        this.isForwards = true;
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    //==========================PUBLIC METHODS==========================//
    @Override
    /*
      Run method from {@link Thread Thread} interface. The method executes the {@link #search()} search} method to add
      edges to the initial graph.
     */
    public void run() {
        this.currentGraph = search();
        pdagToDag(this.currentGraph);
    }

    //===========================PRIVATE METHODS========================//
    /**
     * Greedy equivalence search: Start from the empty graph, add edges till
     * model is significant. Then start deleting edges till a minimum is
     * achieved.
     *
     * @return the resulting Pattern.
     */
    private Graph search() {
        if (!S.isEmpty()) {
            startTime = System.currentTimeMillis();
            numTotalCalls = 0;
            numNonCachedCalls = 0;

            Graph graph = new EdgeListGraph(this.initialDag);

            // Method 1-- original.
            double scoreInitial = scoreDag(graph);

            // Do backward search.
            fes(graph, scoreInitial);

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
        return this.initialDag;
    }

    /**
     * Forward equivalence search.
     *
     * @param graph The graph in the state prior to the forward equivalence
     * search.
     * @param score The score in the state prior to the forward equivalence
     * search
     * @return the score in the state after the FES method. Note that the graph
     * is changed as a side-effect to its state after the forward equivalence
     * search.
     */
    private double fes(Graph graph, double score) {
        //System.out.println("** FORWARD EQUIVALENCE SEARCH");
        double bestScore = score;
        double bestInsert;

        iterations = 0;

        // Calling fs to calculate best edge to add.
        enlaces = S;
        bestInsert = fs(graph);
        while ((x_i != null) && (iterations < this.maxIt)) {
            // Inserting edge
            insert(x_i, y_i, t_0, graph);

            //PDAGtoCPDAG
            if (update)
                updateEdges(graph);
            else
                rebuildPattern(graph);

            bestScore = bestInsert;

            // Checking that the maximum number of edges has not been reached
            if (getMaxNumEdges() != -1 && graph.getNumEdges() >= getMaxNumEdges()) {
                break;
            }

            // Executing FS function to calculate the best edge to be added
            bestInsert = fs(graph);

            // Indicating that the thread has added an edge to the graph
            this.flag = true;
            iterations++;
        }
        return bestScore;

    }

    /**
     * Forward search. Finds the best possible edge to be added into the current
     * graph and returns its score.
     *
     * @param graph The graph in the state prior to the forward equivalence
     * search.
     * @return the score in the state after the forward equivalence search. Note
     * that the graph is changed as a side effect to its state after the forward
     * equivalence search.
     */
    private double fs(Graph graph) {
        x_i = y_i = null;
        t_0 = null;

        Set<EdgeSearch> newScores;
        if (parallel)
            newScores = enlaces.parallelStream()
                .map(e -> scoreEdge(graph, e))
                .collect(Collectors.toSet());
        else
            newScores = enlaces.stream()
                .map(e -> scoreEdge(graph, e))
                .collect(Collectors.toSet());
        
        HashSet<EdgeSearch> temp = new HashSet<>();
        temp.addAll(newScores);
        temp.addAll(this.scores);
        this.scores = temp;
        
        EdgeSearch max = Collections.max(this.scores);

        if (max.score > 0) {
            //Assigning values to x_i, y_i and t_0
            x_i = max.edge.getNode1();
            y_i = max.edge.getNode2();
            t_0 = max.hSubset;

            // Deleting the selected edge from enlaces
            enlaces.remove(max.edge);
            this.scores.remove(max);
            S.remove(max.edge);
        }

        return max.score;
    }

    private void updateEdges(Graph graph){
        // Modo normal
        if (!speedUp) {
            // Getting the common adjacents of x_i and y_i
            Set<Node> process = revertToCPDAG(graph);
            removeEdgesNotNeighbors(graph, process);
        }
        // Modo heurístico. No comprobamos los enlaces invertidos en revertToCPDAG
        else {
            // Getting the common adjacents of x_i and y_i
            Set<Node> process = new HashSet<>();
            removeEdgesNotNeighbors(graph, process);
        }
    }

    private void removeEdgesNotNeighbors(Graph graph, Set<Node> process) {
        process.add(x_i);
        process.add(y_i);

        process.addAll(graph.getAdjacentNodes(x_i));
        process.addAll(graph.getAdjacentNodes(y_i));

        enlaces = new HashSet<>(S);
        enlaces.removeIf(edge -> {
            Node x = edge.getNode1();
            Node y = edge.getNode2();
            return !process.contains(x) && !process.contains(y);
        });
    }

    private Set<Node> revertToCPDAG(Graph graph) {
        GraphSearchUtils.basicCpdag(graph);
        MeekRules rules = new MeekRules();
        return rules.orientImplied(graph);
    }

    private EdgeSearch scoreEdge(Graph graph, Edge edge) {
        Node _x = edge.getNode1();
        Node _y = edge.getNode2();

        if (!graph.isAdjacentTo(_x, _y)) {
            List<Node> tNeighbors = getSubsetOfNeighbors(_x, _y, graph);

            HashSet<Node> tSubset = new HashSet<>();
            double insertEval = insertEval(_x, _y, new HashSet<>(), graph, problem);

            if (insertEval > 0) {
                HashSet<Node> naYX = findNaYX(_x, _y, graph);
                HashSet<Node> naYXT = new HashSet<>(naYX);

                // TESTS 1 AND 2
                if (isClique(naYXT, graph) && isSemiDirectedBlocked(_x, _y, naYXT, graph)) {
                    double greedyScore = insertEval;
                    int bestNodeIndex;
                    Node bestNode = null;

                    do {
                        bestNodeIndex = -1;
                        for (int k = 0; k < tNeighbors.size(); k++) {
                            Node node = tNeighbors.get(k);
                            HashSet<Node> newT = new HashSet<>(tSubset);
                            newT.add(node);
                            insertEval = insertEval(_x, _y, newT, graph, problem);

                            if (insertEval <= greedyScore) {
                                continue;
                            }

                            naYXT = new HashSet<>(newT);
                            naYXT.addAll(naYX);

                            // TESTS 1 AND 2
                            if (!isClique(naYXT, graph) || !isSemiDirectedBlocked(_x, _y, naYXT, graph)) {
                                continue;
                            }

                            bestNodeIndex = k;
                            bestNode = node;
                            greedyScore = insertEval;
                        }
                        if (bestNodeIndex != -1) {
                            tSubset.add(bestNode);
                            tNeighbors.remove(bestNodeIndex);
                        }

                    } while ((bestNodeIndex != -1) && (tSubset.size() <= 1));

                    if (greedyScore > insertEval) {
                        insertEval = greedyScore;
                    }
                    return new EdgeSearch(insertEval, tSubset, edge);

                }
            }
        }
        return new EdgeSearch(0, new HashSet<>(), edge);

    }

}
