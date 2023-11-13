package consensusBN;

import java.util.ArrayList;
import java.util.List;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Node;

import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;

public class ConsensusUnion {

    public static Dag fusionUnion(ArrayList<Dag> dags) {
        ArrayList<Node> alpha = alphaOrder(dags);

        ArrayList<Dag> outputDags = new ArrayList<>();

        for (Dag dag : dags) {
            outputDags.add(transformToAlpha(dag, alpha));
        }

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
}
