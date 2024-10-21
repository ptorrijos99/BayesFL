package consensusBN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;

public class BetaToAlpha {

    public static Dag transformToAlpha(Dag G, List<Node> alfa) {
        HashMap<Node, Integer> alfaHash = new HashMap<>();
        for (int i = 0; i < alfa.size(); i++) {
            Node n = alfa.get(i);
            alfaHash.put(n, i);
        }

        // Construct beta order as close as possible to alfa.
        ArrayList<Node> beta = new ArrayList<>(alfa.size());
        Dag G_aux = new Dag(G);
        HashSet<Node> sinkNodes = new HashSet<>(getSinkNodes(G_aux));

        while (G_aux.getNumNodes() > 0) {
            Node sink = sinkNodes.iterator().next();

            List<Node> pa = G_aux.getParents(sink);
            G_aux.removeNode(sink);
            sinkNodes.remove(sink);

            // Compute the new sink nodes
            for (Node nodep : pa) {
                if (G_aux.getChildren(nodep).isEmpty()) {
                    sinkNodes.add(nodep);
                }
            }

            int i = 0;
            if (!beta.isEmpty()) {
                while (true) {
                    Node nodej = beta.get(i);

                    if (alfaHash.get(nodej) > alfaHash.get(sink)) {
                        break;
                    }
                    if (G.getParents(nodej).contains(sink)) {
                        break;
                    }
                    if (i == beta.size() - 1) {
                        break;
                    }
                    i++;
                }
            }
            beta.add(i, sink);
        }

        // Transform graph G into an I-map minimal with alpha order
        G = new Dag(G);
        ArrayList<Node> aux_beta = new ArrayList<>();
        aux_beta.add(beta.remove(0));

        while (!beta.isEmpty()) { // Check each variable from the sink nodes.
            aux_beta.add(beta.remove(0));
            int i = aux_beta.size() - 1;

            while (i > 0) {
                Node nodeY = aux_beta.get(i);
                Node nodeZ = aux_beta.get(i - 1);

                if ((nodeZ != null) && (alfaHash.get(nodeZ) > alfaHash.get(nodeY))) {
                    if (G.getEdge(nodeZ, nodeY) != null) {
                        List<Node> paZ = G.getParents(nodeZ);
                        List<Node> paY = G.getParents(nodeY);
                        paY.remove(nodeZ);
                        G.removeEdge(nodeZ, nodeY);
                        G.addEdge(new Edge(nodeY, nodeZ, Endpoint.TAIL, Endpoint.ARROW));

                        for (Node nodep : paZ) {
                            if (G.getEdge(nodep, nodeY) == null) {
                                G.addEdge(new Edge(nodep, nodeY, Endpoint.TAIL, Endpoint.ARROW));
                            }
                        }
                        for (Node nodep : paY) {
                            if (G.getEdge(nodep, nodeZ) == null) {
                                G.addEdge(new Edge(nodep, nodeZ, Endpoint.TAIL, Endpoint.ARROW));
                            }
                        }
                    }
                    aux_beta.remove(nodeY);
                    aux_beta.add(i - 1, nodeY);
                    i--;
                } else break;
            }
        }
        return G;
    }

     static ArrayList<Node> getSinkNodes(Dag g) {
        ArrayList<Node> sourcesNodes = new ArrayList<>();

        for (Node node : g.getNodes()) {
            if (g.getChildren(node).isEmpty()) {
                sourcesNodes.add(node);
            }
        }
        return sourcesNodes;
    }
}
