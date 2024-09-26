package consensusBN;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import edu.cmu.tetrad.graph.*;

public class AlphaOrder {

    // Heurística para encontrar un orden de consenso.
    // Se basa en los enlaces que generaría seguir una secuencia creada desde los nodos sumideros hacia arriba.
    public static ArrayList<Node> alphaOrder(List<Dag> dags) {
        ArrayList<Dag> auxDags = new ArrayList<>();
        for (Dag i : dags) {
            Dag aux_G = new Dag(i);
            auxDags.add(aux_G);
        }

        List<Node> nodes = new ArrayList<>(dags.get(0).getNodes());
        LinkedList<Node> computedAlpha = new LinkedList<>();

        while (!nodes.isEmpty()) {
            Node node_alpha = computeNext(auxDags, nodes);
            computedAlpha.addFirst(node_alpha);
            for (Dag g : auxDags) {
                removeNode(g, node_alpha);
            }
            nodes.remove(node_alpha);
        }
        return new ArrayList<>(computedAlpha);
    }

    static Node computeNext(List<Dag> dags, List<Node> nodes) {
        int changes;
        int inversion = 0;
        int addition = 0;
        Node bestNode = null;
        int min = Integer.MAX_VALUE;

        for (Node nodei : nodes) {
            for (Dag g : dags) {
                ArrayList<Edge> inserted = new ArrayList<>();
                List<Node> children = g.getChildren(nodei);
                inversion += (children.size() - 1);
                List<Node> paX = g.getParents(nodei);
                for (Node child : children) {
                    List<Node> paY = g.getParents(child);
                    for (Node nodep : paX) {
                        if (g.getEdge(nodep, child) == null) {
                            addition++;
                        }
                    }
                    for (Node nodec : paY) {
                        if (!nodec.equals(nodei)) {
                            if ((g.getEdge(nodec, nodei) == null) && (g.getEdge(nodei, nodec) == null)) {
                                Edge toBeInserted = new Edge(nodec, nodei, Endpoint.CIRCLE, Endpoint.CIRCLE);
                                boolean contains = false;
                                for (Edge e : inserted) {
                                    if ((e.getNode1().equals(nodec) && (e.getNode2().equals(nodei)))
                                            || ((e.getNode1().equals(nodei) && (e.getNode2().equals(nodec))))) {
                                        contains = true;
                                        break;
                                    }
                                }
                                if (!contains) {
                                    addition++;
                                    inserted.add(toBeInserted);
                                }
                            }
                        }
                    }
                }
            }
            changes = inversion + addition;
            if (changes < min) {
                min = changes;
                bestNode = nodei;
            }
            inversion = 0;
            addition = 0;
        }
        return bestNode;
    }

    static void removeNode(Dag g, Node node_alpha) {
        Node node_alpha_g = g.getNode(node_alpha.getName());

        List<Node> children = g.getChildren(node_alpha_g);

        while (!children.isEmpty()) {
            int i = 0;
            Node child;
            boolean seguir;
            do {
                child = children.get(i++);
                g.removeEdge(node_alpha_g, child);
                seguir = false;
                if (g.paths().existsDirectedPath(node_alpha_g, child)) {
                    seguir = true;
                    g.addEdge(new Edge(node_alpha_g, child, Endpoint.TAIL, Endpoint.ARROW));
                }
            } while (seguir);

            List<Node> paX = g.getParents(node_alpha_g);
            List<Node> paY = g.getParents(child);
            paY.remove(node_alpha_g);
            g.addEdge(new Edge(child, node_alpha_g, Endpoint.TAIL, Endpoint.ARROW));
            for (Node nodep : paX) {
                Edge pay = g.getEdge(nodep, child);
                if (pay == null) {
                    g.addEdge(new Edge(nodep, child, Endpoint.TAIL, Endpoint.ARROW));
                }

            }
            for (Node nodep : paY) {
                Edge paz = g.getEdge(nodep, node_alpha_g);
                if (paz == null) {
                    g.addEdge(new Edge(nodep, node_alpha_g, Endpoint.TAIL, Endpoint.ARROW));
                }
            }

            children.remove(child);
        }
        g.removeNode(node_alpha_g);
    }

}
