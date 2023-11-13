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
        return fusionUnion(dags, "union");
    }

    /**
     * Union of the DAGs with possibility of limiting the total edges, previously transformed to the same alpha order.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public static Dag fusionUnion(ArrayList<Dag> dags, String method) {
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

        switch (method) {
            // Option 1: Add the edges in order of frequency, until the mean number of edges of the DAGs is reached
            case "MeanEdgesLimit":
                int meanEdges = 0;
                for (Dag d : dags) {
                    meanEdges += d.getNumEdges() / dags.size();
                }
                return applyNumberEdgesLimit(alpha, edgeFrequency, edges, meanEdges);
            // Option 2: Add the edges in order of frequency, until the max number of edges of the DAGs is reached
            case "MaxEdgesLimit":
                int maxEdges = 0;
                for (Dag d : dags) {
                    maxEdges = Math.max(maxEdges, d.getNumEdges());
                }
                return applyNumberEdgesLimit(alpha, edgeFrequency, edges, maxEdges);
            // Option 3: Add the edges in order of frequency, until the min number of edges of the DAGs is reached
            case "MinEdgesLimit":
                int minEdges = Integer.MAX_VALUE;
                for (Dag d : dags) {
                    minEdges = Math.min(minEdges, d.getNumEdges());
                }
                return applyNumberEdgesLimit(alpha, edgeFrequency, edges, minEdges);
            // Option 4: Add the edges in order of frequency, until the frequency of the edges is less than the mean
            case "FrequencyMeanLimit":
                int meanFrequency = 0;
                for (Integer frequency : edgeFrequency.values()) {
                    meanFrequency += frequency / edgeFrequency.size();
                }
                return applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, meanFrequency);
            // Option 5: Add the edges in order of frequency, until the frequency of the edges is more than 1
            case "FrequencyOneLimit":
                return applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, 1);
            // Option 6: Add the edges in order of frequency, until the frequency of the edges is more than 2
            case "FrequencyTwoLimit":
                return applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, 2);
            // Option 7: Add the edges in order of frequency, with the consensus of 90% of the DAGs
            case "Consensus90":
                return applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, (int) Math.ceil(dags.size() * 0.9));
            // Option 8: Add the edges in order of frequency, with the consensus of 80% of the DAGs
            case "Consensus80":
                return applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, (int) Math.ceil(dags.size() * 0.8));
            // Option 9: Add the edges in order of frequency, with the consensus of 50% of the DAGs
            case "Consensus50":
                return applyEdgesFrequencyLimit(alpha, edgeFrequency, edges, (int) Math.ceil(dags.size() * 0.5));
            // Option 10: Add the edges in order of frequency, limiting the maximum number of parents of each node to 2
            case "MaxParentsTwo":
                return applyMaxParents(alpha, edges, 2);
            // Option 11: Add the edges in order of frequency, limiting the maximum number of parents of each node to 3
            case "MaxParentsThree":
                return applyMaxParents(alpha, edges, 3);
        }

        return applyUnion(alpha, outputDags);
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
    private static Dag applyNumberEdgesLimit(ArrayList<Node> alpha, HashMap<Edge, Integer> edgeFrequency, List<Edge> edges, int limit) {
        Dag union = new Dag(alpha);
        int numEdges = 0;
        for (Edge edge : edges) {
            if (edgeFrequency.get(edge) > limit) {
                union.addEdge(edge);
            }
            numEdges++;
            if (numEdges >= limit) {
                break;
            }
        }
        return union;
    }

    // Add the edges in order of frequency, until the frequency of the edges is less than a number
    private static Dag applyEdgesFrequencyLimit(ArrayList<Node> alpha, HashMap<Edge, Integer> edgeFrequency, List<Edge> edges, int limit) {
        Dag union = new Dag(alpha);
        for (Edge edge : edges) {
            if (edgeFrequency.get(edge) > limit) {
                union.addEdge(edge);
            }
        }
        return union;
    }

    // Add the edges in order of frequency, limiting the maximum number of parents of each node
    private static Dag applyMaxParents(ArrayList<Node> alpha, List<Edge> edges, int maxParents) {
        Dag union = new Dag(alpha);
        for (Edge edge : edges) {
            Node child;
            if (edge.getEndpoint1() == Endpoint.TAIL) {
                child = edge.getNode1();
            } else {
                child = edge.getNode2();
            }

            if (union.getParents(child).size() < maxParents) {
                union.addEdge(edge);
            }
        }
        return union;
    }


}
