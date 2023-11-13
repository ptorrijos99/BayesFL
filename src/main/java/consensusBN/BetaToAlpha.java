package consensusBN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;

public class BetaToAlpha {

    public static Dag transform(Dag G, ArrayList<Node> alfa) {
        HashMap<Node, Integer> alfaHash = new HashMap<>();
        for (int i = 0; i < alfa.size(); i++) {
            Node n = alfa.get(i);
            alfaHash.put(n, i);
        }

        ArrayList<Node> beta = new ArrayList<>();
        Dag G_aux = new Dag(G);

        ArrayList<Node> sinkNodes = getSinkNodes(G_aux);
        beta.add(sinkNodes.get(0));
        List<Node> pa = G_aux.getParents(sinkNodes.get(0));
        G_aux.removeNode(sinkNodes.get(0));
        sinkNodes.remove(0);
        
        // Compute the new sink nodes
        for (Node nodep : pa) {
            List<Node> chld = G_aux.getChildren(nodep);
            if (chld.isEmpty()) {
                sinkNodes.add(nodep);
            }
        }

        // Construct beta order as closer as possible to alfa.
        while (G_aux.getNumNodes() > 0) {
            Node sink = sinkNodes.get(0);
            pa = G_aux.getParents(sink);
            G_aux.removeNode(sink);
            sinkNodes.remove(0);
            // Compute the new sink nodes
            for (Node nodep : pa) {
                List<Node> chld = G_aux.getChildren(nodep);
                if (chld.isEmpty()) {
                    sinkNodes.add(nodep);
                }
            }

            int index_alfa_sink = alfaHash.get(sink);
            int i = 0;

            while (true) {
                Node nodej = beta.get(i);
                int index_alfa_nodej = alfaHash.get(nodej);

                if (index_alfa_nodej > index_alfa_sink) {
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

            beta.add(i, sink);
        }

        // transform graph G into an I-map minimal with alpha order
        ArrayList<Node> aux_beta = new ArrayList<>();
        aux_beta.add(beta.get(0));
        beta.remove(0);

        while (!beta.isEmpty()) { // check each variable from the sink nodes.

            aux_beta.add(beta.get(0));
            beta.remove(0);
            int i = aux_beta.size();
            boolean ok = true;

            while (ok) {
                if (i == 1) {
                    break;
                }
                ok = false;
                Node nodeY = aux_beta.get(i - 1);
                Node nodeZ = aux_beta.get(i - 2);

                if ((nodeZ != null) && (alfaHash.get(nodeZ) > alfaHash.get(nodeY))) {
                    if (G.getEdge(nodeZ, nodeY) != null) {
                        List<Node> paZ = G.getParents(nodeZ);
                        List<Node> paY = G.getParents(nodeY);
                        paY.remove(nodeZ);
                        G.removeEdge(nodeZ, nodeY);
                        G.addEdge(new Edge(nodeY, nodeZ, Endpoint.TAIL, Endpoint.ARROW));
                        for (Node nodep : paZ) {
                            Edge pay = G.getEdge(nodep, nodeY);
                            if (pay == null) {
                                G.addEdge(new Edge(nodep, nodeY, Endpoint.TAIL, Endpoint.ARROW));
                            }
                        }
                        for (Node nodep : paY) {
                            Edge paz = G.getEdge(nodep, nodeZ);
                            if (paz == null) {
                                G.addEdge(new Edge(nodep, nodeZ, Endpoint.TAIL, Endpoint.ARROW));
                            }
                        }
                    }
                    ok = true;
                    aux_beta.remove(nodeY);
                    aux_beta.add(i - 2, nodeY);
                    i--;
                }
            }
        }
        beta = aux_beta;
        return G;
    }

     static ArrayList<Node> getSinkNodes(Dag g) {
        ArrayList<Node> sourcesNodes = new ArrayList<>();
        List<Node> nodes = g.getNodes();

        for (Node nodei : nodes) {
            if (g.getChildren(nodei).isEmpty()) {
                sourcesNodes.add(nodei);
            }
        }
        return sourcesNodes;
    }
}
