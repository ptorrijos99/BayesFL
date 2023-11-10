package consensusBN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;

public class BetaToAlpha {

    Dag G;
    ArrayList<Node> beta;
    ArrayList<Node> alfa;
    HashMap<Node, Integer> alfaHash = new HashMap<>();
    Dag G_aux = null;
    int numberOfInsertedEdges = 0;

    public BetaToAlpha(Dag G, ArrayList<Node> alfa) {
        this.alfa = alfa;
        this.G = G;
        this.beta = null;
        for (int i = 0; i < alfa.size(); i++) {
            Node n = alfa.get(i);
            alfaHash.put(n, i);
        }
    }

    public void transform() {
        this.G_aux = new Dag(this.G);
        this.beta = new ArrayList<>();
        ArrayList<Node> sinkNodes = getSinkNodes(this.G_aux);
        this.beta.add(sinkNodes.get(0));
        List<Node> pa = G_aux.getParents(sinkNodes.get(0));
        this.G_aux.removeNode(sinkNodes.get(0));
        sinkNodes.remove(0);
        
        // Compute the new sink nodes
        for (Node nodep : pa) {
            List<Node> chld = G_aux.getChildren(nodep);
            if (chld.isEmpty()) {
                sinkNodes.add(nodep);
            }
        }

        // Construct beta order as closer as possible to alfa.
        while (this.G_aux.getNumNodes() > 0) {
            Node sink = sinkNodes.get(0);
            pa = G_aux.getParents(sink);
            this.G_aux.removeNode(sink);
            sinkNodes.remove(0);
            // Compute the new sink nodes
            for (Node nodep : pa) {
                List<Node> chld = G_aux.getChildren(nodep);
                if (chld.isEmpty()) {
                    sinkNodes.add(nodep);
                }
            }

            int index_alfa_sink = this.alfaHash.get(sink);
            int i = 0;

            while (true) {
                Node nodej = this.beta.get(i);
                int index_alfa_nodej = this.alfaHash.get(nodej);

                if (index_alfa_nodej > index_alfa_sink) {
                    break;
                }
                if (this.G.getParents(nodej).contains(sink)) {
                    break;
                }
                if (i == this.beta.size() - 1) {
                    break;
                }
                i++;
            }

            this.beta.add(i, sink);
        }

        // transform graph G into an I-map minimal with alpha order
        ArrayList<Node> aux_beta = new ArrayList<>();
        aux_beta.add(this.beta.get(0));
        this.beta.remove(0);

        while (!this.beta.isEmpty()) { // check each variable from the sink nodes.

            aux_beta.add(this.beta.get(0));
            this.beta.remove(0);
            int i = aux_beta.size();
            boolean ok = true;

            while (ok) {
                if (i == 1) {
                    break;
                }
                ok = false;
                Node nodeY = aux_beta.get(i - 1);
                Node nodeZ = aux_beta.get(i - 2);

                if ((nodeZ != null) && (this.alfaHash.get(nodeZ) > this.alfaHash.get(nodeY))) {
                    if (this.G.getEdge(nodeZ, nodeY) != null) {
                        List<Node> paZ = this.G.getParents(nodeZ);
                        List<Node> paY = this.G.getParents(nodeY);
                        paY.remove(nodeZ);
                        this.G.removeEdge(nodeZ, nodeY);
                        this.G.addEdge(new Edge(nodeY, nodeZ, Endpoint.TAIL, Endpoint.ARROW));
                        for (Node nodep : paZ) {
                            Edge pay = this.G.getEdge(nodep, nodeY);
                            if (pay == null) {
                                this.G.addEdge(new Edge(nodep, nodeY, Endpoint.TAIL, Endpoint.ARROW));
                                this.numberOfInsertedEdges++;
                            }
                        }
                        for (Node nodep : paY) {
                            Edge paz = this.G.getEdge(nodep, nodeZ);
                            if (paz == null) {
                                this.G.addEdge(new Edge(nodep, nodeZ, Endpoint.TAIL, Endpoint.ARROW));
                                this.numberOfInsertedEdges++;
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
        this.beta = aux_beta;
    }

    public int getNumberOfInsertedEdges() {
        return this.numberOfInsertedEdges;
    }

    ArrayList<Node> getSinkNodes(Dag g) {
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
