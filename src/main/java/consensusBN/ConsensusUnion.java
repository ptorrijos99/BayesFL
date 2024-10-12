package consensusBN;

import java.util.*;

import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import org.albacete.simd.utils.Utils;

import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;
import static org.albacete.simd.utils.Utils.getConnectedComponent;

public class ConsensusUnion {

    public static Dag initialDag;
    public static boolean allPossibleArcs = false;
    public static boolean metricAgainstOriginalDAGs = true;
    public static boolean metricSMHD = false;

    /**
     * Complete union of the DAGs, previously transformed to the same alpha order.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public static Dag fusionUnion(List<Dag> dags) {
        return fusionUnion(dags, "Union", "0");
    }

    /**
     * Union of the DAGs with possibility of limiting the total edges, previously transformed to the same alpha order.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public static Dag fusionUnion(List<Dag> dags, String method, String limit) {
        switch (method) {
            case "GeneticTW" -> {
                return new GeneticTreeWidthUnion(dags, 42, Integer.parseInt(limit)).fusionUnion();
            }
            case "MaxTreewidthBefore" -> {
                return applyGreedyMaxTreewidthBefore(dags, limit);
            }
            case "MaxTreewidthBeforeWoRepeat" -> {
                return applyGreedyMaxTreewidthBeforeWoRepeat(dags, limit);
            }
        }

        ArrayList<Node> alpha = alphaOrder(dags);

        ArrayList<Dag> outputDags = new ArrayList<>();
        for (Dag dag : dags) {
            outputDags.add(transformToAlpha(dag, alpha));
        }

        // Option 0: Total union of the DAGs
        if (method.equals("Union")) {
            return applyUnion(alpha, outputDags);
        }

        // Order the edges of all the DAGs by the number of times they appear
        HashMap<Edge, Integer> edgeFrequency = new HashMap<>();

        for (Dag d : outputDags) {
            for (Edge edge : d.getEdges()) {
                edgeFrequency.put(edge, edgeFrequency.getOrDefault(edge, 0) + 1);
            }
        }

        List<Edge> edges = new ArrayList<>(edgeFrequency.keySet());
        edges.sort((o1, o2) -> edgeFrequency.get(o2) - edgeFrequency.get(o1));

        return switch (method) {
            // Option 1: Add the edges in order of frequency, until the maximum number of edges of the DAGs is reached
            case "MaxEdges" -> applyEdgesLimit(alpha, dags, edges, limit);
            // Option 2: Add the edges in order of frequency, until the frequency of the edges is more than a number
            case "MaxFrequency" -> applyFrequencyLimit(alpha, edgeFrequency, edges, limit);
            // Option 3: Add the edges in order of frequency, limiting the maximum number of parents of each node
            case "MaxParents" -> applyMaxParents(alpha, edges, limit);
            // Option 4: Add the edges in order of frequency, limiting the maximum treewidth
            case "MaxTreewidth" -> applyGreedyMaxTreewidth(alpha, edges, limit);
            // Option 5: Search for the union that maximizes the similarity with the original fusion, limiting the maximum treewidth
            case "SuperGreedyMaxTreewidth" -> applySuperGreedyMaxTreewidth(initialDag, alpha, dags, limit);
            // Default: Total union of the DAGs
            default -> applyUnion(alpha, outputDags);
        };
    }

    // Add all the edges
    public static Dag applyUnion(List<Node> alpha, List<Dag> dags) {
        Dag union = new Dag(alpha);
        for (Node node : alpha) {
            for (Dag d : dags) {
                List<Node> parent = d.getParents(node);
                for (Node pa : parent) {
                    if (!union.isParentOf(pa, node)) {
                        union.addEdge(new Edge(pa, node, Endpoint.TAIL, Endpoint.ARROW));
                    }
                }
            }
        }
        return union;
    }

    // Add the edges in order of frequency, until the number of edges is less than a number
    private static Dag applyEdgesLimit(List<Node> alpha, List<Dag> dags, List<Edge> edges, String limit) {
        int lim = 0;
        switch (limit) {
            case "Mean" -> {
                for (Dag d : dags) {
                    lim += d.getNumEdges();
                }
                lim /= dags.size();
            }
            case "Max" -> {
                for (Dag d : dags) {
                    lim = Math.max(lim, d.getNumEdges());
                }
            }
            case "Min" -> {
                lim = Integer.MAX_VALUE;
                for (Dag d : dags) {
                    lim = Math.min(lim, d.getNumEdges());
                }
            }
            default -> lim = Integer.parseInt(limit);
        }

        Dag union = new Dag(alpha);
        int numEdges = 0;
        for (Edge edge : edges) {
            if (numEdges >= lim) {
                break;
            }
            union.addEdge(edge);
            numEdges++;
        }
        return union;
    }

    // Add the edges in order of frequency, until the frequency of the edges is less or equal than a number
    private static Dag applyFrequencyLimit(List<Node> alpha, HashMap<Edge, Integer> edgeFrequency, List<Edge> edges, String limit) {
        int lim = 0;
        if (limit.equals("Mean")) {
            for (Integer frequency : edgeFrequency.values()) {
                lim += frequency;
            }
            lim /= edgeFrequency.size();
        } else {
            lim = Integer.parseInt(limit);
        }

        Dag union = new Dag(alpha);
        for (Edge edge : edges) {
            if (edgeFrequency.get(edge) >= lim) {
                union.addEdge(edge);
            }
        }
        return union;
    }

    // Add the edges in order of frequency, limiting the maximum number of parents of each node
    private static Dag applyMaxParents(List<Node> alpha, List<Edge> edges, String maxParents) {
        int lim = Integer.parseInt(maxParents);
        Dag union = new Dag(alpha);
        for (Edge edge : edges) {
            Node child;
            if (edge.getEndpoint1() == Endpoint.TAIL) {
                child = edge.getNode1();
            } else {
                child = edge.getNode2();
            }

            if (union.getParents(child).size() < lim) {
                union.addEdge(edge);
            }
        }
        return union;
    }

    // Add the edges in order of frequency, limiting the maximum tree width. Greedy algorithm.
    public static Dag applyGreedyMaxTreewidth(List<Node> alpha, List<Edge> edges, String maxTreewidth) {
        int maxCliqueSize = Integer.parseInt(maxTreewidth);
        Dag union = new Dag(alpha);

        for (Edge edge : edges) {
            // Add the edge
            union.addEdge(edge);

            // Get the cliques of the union
            Map<Node, Set<Node>> cliques = Utils.getMoralTriangulatedCliques(union);

            // If the maximum clique size is greater than the limit, remove the edge
            for (Set<Node> clique : cliques.values()) {
                if (clique.size() > maxCliqueSize) {
                    union.removeEdge(edge);
                    break;
                }
            }
        }
        return union;
    }

    // Add the edges in order of frequency, limiting the maximum tree width. Try to add the edges on the original DAGs,
    // instead of in the union, to obtain a union that respect the limit of treewidth. Do not repeat the edges that
    // appears in more than one original DAG. Greedy algorithm.
    public static Dag applyGreedyMaxTreewidthBeforeWoRepeat(List<Dag> dags, String maxTreewidth) {
        List<Node> alpha = alphaOrder(dags);

        List<Dag> newDags = originalDAGsGreedyTreewidthBeforeWoRepeat(dags, alpha, maxTreewidth);

        List<Dag> transformedDags = new ArrayList<>();
        for (Dag dag : newDags) {
            transformedDags.add(transformToAlpha(dag, alpha));
        }

        return applyUnion(alpha, transformedDags);
    }

    public static List<Dag> originalDAGsGreedyTreewidthBeforeWoRepeat(List<Dag> dags, List<Node> alpha, String maxTreewidth) {
        int maxCliqueSize = Integer.parseInt(maxTreewidth);

        ArrayList<Dag> outputDags = new ArrayList<>();
        for (Dag ignored : dags) {
            outputDags.add(new Dag(dags.get(0).getNodes()));
        }

        // Order the edges of all the DAGs by the number of times they appear
        HashMap<Edge, Integer> edgeFrequency = new HashMap<>();
        HashMap<Edge, List<Integer>> edgeDag = new HashMap<>();

        for (Dag d : dags) {
            for (Edge edge : d.getEdges()) {
                edgeFrequency.put(edge, edgeFrequency.getOrDefault(edge, 0) + 1);
                edgeDag.putIfAbsent(edge, new ArrayList<>());
                edgeDag.get(edge).add(dags.indexOf(d));
            }
        }

        List<Edge> edges = new ArrayList<>(edgeFrequency.keySet());
        edges.sort((o1, o2) -> edgeFrequency.get(o2) - edgeFrequency.get(o1));

        for (Edge edge : edges) {
            // Add the edge to the DAGs that have it initially
            for (int dagIndex : edgeDag.get(edge)) {
                outputDags.get(dagIndex).addEdge(edge);
            }

            List<Dag> transformedDags = new ArrayList<>();
            for (Dag dag : outputDags) {
                transformedDags.add(transformToAlpha(dag, alpha));
            }
            // Get the union fusion
            Dag union = fusionUnion(transformedDags);

            // Get the cliques of the union
            Map<Node, Set<Node>> cliques = Utils.getMoralTriangulatedCliques(union);

            // If the maximum clique size is greater than the limit, remove the edge
            for (Set<Node> clique : cliques.values()) {
                if (clique.size() > maxCliqueSize) {
                    for (int dagIndex : edgeDag.get(edge)) {
                        outputDags.get(dagIndex).removeEdge(edge);
                    }
                    break;
                }
            }
        }

        return outputDags;
    }


    // Add the edges in order of frequency, limiting the maximum tree width. Try to add the edges on the original DAGs,
    // instead of in the union, to obtain a union that respect the limit of treewidth. Greedy algorithm.
    public static Dag applyGreedyMaxTreewidthBefore(List<Dag> dags, String maxTreewidth) {
        List<Node> alpha = alphaOrder(dags);

        List<Dag> newDags = originalDAGsGreedyTreewidthBefore(dags, alpha, maxTreewidth);

        List<Dag> transformedDags = new ArrayList<>();
        for (Dag dag : newDags) {
            transformedDags.add(transformToAlpha(dag, alpha));
        }

        return applyUnion(alpha, transformedDags);
    }

    public static List<Dag> originalDAGsGreedyTreewidthBefore(List<Dag> dags, List<Node> alpha, String maxTreewidth) {
        int maxCliqueSize = Integer.parseInt(maxTreewidth);

        ArrayList<Dag> outputDags = new ArrayList<>();
        List<List<Edge>> edges = new ArrayList<>();
        for (Dag dag : dags) {
            outputDags.add(new Dag(dags.get(0).getNodes()));
            edges.add(new LinkedList<>(dag.getEdges()));
        }

        // While all lists are not empty, try to add a new edge, each time from a different list, while respecting the limit
        while (edges.stream().anyMatch(list -> !list.isEmpty())) {
            for (int i = 0; i < edges.size(); i++) {
                if (edges.get(i).isEmpty()) continue;

                Edge edge = edges.get(i).remove(0);
                outputDags.get(i).addEdge(edge);

                // Get the union fusion
                List<Dag> transformedDags = new ArrayList<>();
                for (Dag dag : outputDags) {
                    transformedDags.add(transformToAlpha(dag, alpha));
                }
                Dag union = fusionUnion(transformedDags);

                // Check the treewidth
                Map<Node, Set<Node>> cliques = Utils.getMoralTriangulatedCliques(union);
                for (Set<Node> clique : cliques.values()) {
                    // If the treewidth is greather than the limit, remove the edge
                    if (clique.size() > maxCliqueSize) {
                        outputDags.get(i).removeEdge(edge);
                        break;
                    }
                }
            }
        }

        return outputDags;
    }

    // GES-like algorithm. In each iteration, adds, removes or reverses the edge that maximizes the score while the tw
    // is less than the limit. Stops when no edge can be added, removed or reversed without decreasing the score.
    private static Dag applySuperGreedyMaxTreewidth(Dag initialDag, List<Node> alpha, List<Dag> dags, String maxTreewidth) {
        double mejora = 0;
        Set<Node> connectedComponent = new HashSet<>(alpha);
        int maxCliqueSize = Integer.parseInt(maxTreewidth);

        Dag finalDag;
        Set<Edge> includedArcs;
        if (initialDag == null) {
            finalDag = new Dag(alpha);
            includedArcs = new HashSet<>();
        } else{
            finalDag = new Dag(initialDag);
            includedArcs = finalDag.getEdges();
        }

        // Get the union fusion
        Dag union = fusionUnion(dags);

        double bestScore;
        Graph moralizedUnion = null;
        List<Graph> moralizedInitialDAGs = null;
        if (!metricAgainstOriginalDAGs) {
            moralizedUnion = Utils.moralize(union);
            bestScore = Utils.SMHDwithoutMoralize(moralizedUnion, Utils.moralize(finalDag));
        } else {
            if (metricSMHD) {
                moralizedInitialDAGs = new ArrayList<>();
                for (Dag dag : dags) {
                    moralizedInitialDAGs.add(Utils.moralize(dag));
                }
                bestScore = Utils.SMHDwithoutMoralize(Utils.moralize(finalDag), moralizedInitialDAGs);
            } else {
                bestScore = Utils.fusionSimilarity(finalDag, dags);
            }
        }

        Set<Edge> notIncludedArcs;
        if (allPossibleArcs) {
            notIncludedArcs = Utils.calculateArcs(alpha);
            notIncludedArcs.removeAll(includedArcs);
        }
        else {
            notIncludedArcs = union.getEdges();
            notIncludedArcs.removeAll(includedArcs);
        }

        // Crear un array de resultados
        Result[] results = new Result[includedArcs.size() + notIncludedArcs.size()];

        // Crear un array de booleanos para saber qué enlace está y cual no, incluyendo los enlaces incluidos y los no
        boolean[] included = new boolean[includedArcs.size() + notIncludedArcs.size()];
        ArrayList<Edge> allArcs = new ArrayList<>(includedArcs);
        allArcs.addAll(notIncludedArcs);
        for (int i = 0; i < includedArcs.size(); i++) {
            included[i] = true;
        }

        // While there is improvement in the score with less treewidth than the limit
        while (true) {
            Edge iterationEdge = null;

            // Evaluate operations (add, reverse, remove arcs)
            for (int i = 0; i < included.length; i++) {
                Edge arc = allArcs.get(i);
                if (connectedComponent.contains(arc.getNode1()) || connectedComponent.contains(arc.getNode2())) {
                    if (included[i]) {
                        // TODO: DEMOSTRAR QUE NUNCA SE PUEDE ELIMINAR UN ENLACE
                        results[i] = tryArc(moralizedUnion, moralizedInitialDAGs, dags, finalDag, arc, maxCliqueSize, 1, i);

                        // TODO: DEMOSTRAR QUE NUNCA SE PUEDE REVERTIR UN ENLACE
                        /*if (notIncludedArcs.contains(new Edge(arc.getNode2(), arc.getNode1(), Endpoint.TAIL, Endpoint.ARROW))) {
                            Result op2 = tryArc(moralizedUnion, moralizedInitialDAGs, dags, finalDag, arc, maxCliqueSize, 2, i);
                            if (results[i] == null || (op2 != null && op2.score < results[i].score)) {
                                results[i] = op2;
                            }
                        }*/
                    } else {
                        results[i] = tryArc(moralizedUnion, moralizedInitialDAGs, dags, finalDag, arc, maxCliqueSize, 0, i);
                    }
                } else if (results[i] != null) {
                    results[i].score = results[i].score - mejora;
                }
            }

            // Update the variables
            double lastScore = bestScore;
            int operation = -1;
            int id = -1;
            for (Result result : results) {
                if (result != null && result.score < bestScore) {
                    bestScore = result.score;
                    iterationEdge = allArcs.get(result.id);
                    operation = result.operation;
                    id = result.id;
                }
            }
            mejora = lastScore - bestScore;

            // If no operation improves the score, stop
            if (operation == -1) break;

            switch (operation) {
                case 0:  // Add arc
                    finalDag.addEdge(iterationEdge);
                    notIncludedArcs.remove(iterationEdge);
                    includedArcs.add(iterationEdge);
                    included[id] = true;
                    break;
                case 1:  // Remove arc
                    finalDag.removeEdge(iterationEdge);
                    includedArcs.remove(iterationEdge);
                    notIncludedArcs.add(iterationEdge);
                    included[id] = false;
                    break;
                case 2:  // Reverse arc
                    Edge reversedEdge = new Edge(iterationEdge.getNode2(), iterationEdge.getNode1(), Endpoint.TAIL, Endpoint.ARROW);

                    finalDag.removeEdge(iterationEdge);
                    finalDag.addEdge(reversedEdge);

                    includedArcs.remove(iterationEdge);
                    notIncludedArcs.add(iterationEdge);
                    included[id] = false;

                    includedArcs.add(reversedEdge);
                    notIncludedArcs.remove(reversedEdge);
                    int idReversed = allArcs.indexOf(reversedEdge);
                    included[idReversed] = true;
                    break;
            }

            // Obtener la componente conexa del nuevo enlace incluido
            connectedComponent.clear();
            Node node = iterationEdge.getNode1();
            Graph moralizedGraph = Utils.moralize(finalDag);
            connectedComponent.addAll(getConnectedComponent(moralizedGraph, node));
        }

        return finalDag;
    }

    private static Result tryArc(Graph moralizedUnion, List<Graph> moralizedInitialDAGs, List<Dag> initialDAGs, Dag dag, Edge arc, int maxCliqueSize, int operation, int id) {
        Result score;

        if (operation == 0) {  // Add arc
            // Check that the arc does not create a cycle (edge X -> Y, check that the path from Y to X is not possible)
            if (dag.paths().existsDirectedPath(arc.getNode2(), arc.getNode1())) return null;

            dag.addEdge(arc);

            // Verify the treewidth
            for (Set<Node> clique : Utils.getMoralTriangulatedCliques(dag).values()) {
                if (clique.size() > maxCliqueSize) {
                    dag.removeEdge(arc);
                    return null;
                }
            }

            if (metricAgainstOriginalDAGs) {
                if (metricSMHD) {
                    score = new Result(id, Utils.SMHDwithoutMoralize(Utils.moralize(dag), moralizedInitialDAGs), 0);
                } else {
                    score = new Result(id, Utils.fusionSimilarity(dag, initialDAGs), 0);
                }
            } else {
                score = new Result(id, Utils.SMHDwithoutMoralize(moralizedUnion, Utils.moralize(dag)), 0);
            }
            dag.removeEdge(arc);
        } else if (operation == 1) {  // Remove arc
            dag.removeEdge(arc);
            if (metricAgainstOriginalDAGs) {
                if (metricSMHD) {
                    score = new Result(id, Utils.SMHDwithoutMoralize(Utils.moralize(dag), moralizedInitialDAGs), 1);
                } else {
                    score = new Result(id, Utils.fusionSimilarity(dag, initialDAGs), 1);
                }
            } else {
                score = new Result(id, Utils.SMHDwithoutMoralize(moralizedUnion, Utils.moralize(dag)), 1);// No treewidth or cycle calculation needed, only can be less or equal than the previous
            }
            dag.addEdge(arc);
        } else {  // Reverse arc
            dag.removeEdge(arc);

            // Check that the arc does not create a cycle (edge X -> Y, check that the path from Y to X is not possible)
            if (dag.paths().existsDirectedPath(arc.getNode2(), arc.getNode1())) {
                dag.addEdge(arc);
                return null;
            }

            Edge newEdge = new Edge(arc.getNode2(), arc.getNode1(), Endpoint.TAIL, Endpoint.ARROW);
            dag.addEdge(newEdge);

            // Verify the treewidth
            for (Set<Node> clique : Utils.getMoralTriangulatedCliques(dag).values()) {
                if (clique.size() > maxCliqueSize) {
                    dag.removeEdge(newEdge);
                    dag.addEdge(arc);
                    return null;
                }
            }

            if (metricAgainstOriginalDAGs) {
                if (metricSMHD) {
                    score = new Result(id, Utils.SMHDwithoutMoralize(Utils.moralize(dag), moralizedInitialDAGs), 2);
                } else {
                    score = new Result(id, Utils.fusionSimilarity(dag, initialDAGs), 2);
                }
            } else {
                score = new Result(id, Utils.SMHDwithoutMoralize(moralizedUnion, Utils.moralize(dag)), 2);
            }
            dag.removeEdge(newEdge);
            dag.addEdge(arc);
        }

        return score;
    }

    // Create a class that saves the result
    static class Result {
        public int id;
        public double score;
        public int operation;

        public Result(int id, double score, int operation) {
            this.id = id;
            this.score = score;
            this.operation = operation;
        }
    }
}














