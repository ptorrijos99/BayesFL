package consensusBN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Node;

import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;

public class ConsensusUnion {

    /**
     * Complete union of the DAGs, previously transformed to the same alpha order.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public static Dag fusionUnion(ArrayList<Dag> dags) {
        return fusionUnion(dags, "Union", "0");
    }

    /**
     * Union of the DAGs with possibility of limiting the total edges, previously transformed to the same alpha order.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public static Dag fusionUnion(ArrayList<Dag> dags, String method, String limit) {
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
            case "MaxEdges" -> applyNumberEdgesLimit(alpha, dags, edges, limit);
            // Option 2: Add the edges in order of frequency, until the frequency of the edges is more than a number
            case "MaxFrequency" -> applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, limit);
            // Option 3: Add the edges in order of frequency, limiting the maximum number of parents of each node
            case "MaxParents" -> applyMaxParents(alpha, edges, limit);
            // Default: Total union of the DAGs
            default -> applyUnion(alpha, outputDags);
        };
    }

    // Add all the edges
    private static Dag applyUnion(ArrayList<Node> alpha, ArrayList<Dag> outputDags) {
        Dag union = new Dag(alpha);
        for (Node nodei : alpha) {
            for (Dag d : outputDags) {
                List<Node> parent = d.getParents(nodei);
                for (Node pa : parent) {
                    if (!union.isParentOf(pa, nodei)) {
                        union.addEdge(new Edge(pa, nodei, Endpoint.TAIL, Endpoint.ARROW));
                    }
                }
            }
        }
        return union;
    }

    // Add the edges in order of frequency, until the number of edges is less than a number
    private static Dag applyNumberEdgesLimit(ArrayList<Node> alpha, ArrayList<Dag> dags, List<Edge> edges, String limit) {
        int lim = 0;
        if (limit.equals("Mean")) {
            for (Dag d : dags) {
                lim += d.getNumEdges();
            }
            lim /= dags.size();
        } else if (limit.equals("Max")) {
            for (Dag d : dags) {
                lim = Math.max(lim, d.getNumEdges());
            }
        } else if (limit.equals("Min")) {
            lim = Integer.MAX_VALUE;
            for (Dag d : dags) {
                lim = Math.min(lim, d.getNumEdges());
            }
        } else {
            lim = Integer.parseInt(limit);
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
    private static Dag applyEdgesFrequencyLimit(ArrayList<Node> alpha, HashMap<Edge, Integer> edgeFrequency, List<Edge> edges, String limit) {
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
    private static Dag applyMaxParents(ArrayList<Node> alpha, List<Edge> edges, String maxParents) {
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
}
