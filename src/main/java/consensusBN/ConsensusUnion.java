package consensusBN;

import java.util.ArrayList;
import java.util.List;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Node;

public class ConsensusUnion {

    ArrayList<Node> alpha = null;
    AlphaOrder heuristic;
    ArrayList<Dag> setOfdags;
    Dag union = null;
    int numberOfInsertedEdges = 0;

    public ConsensusUnion(ArrayList<Dag> dags) {
        this.setOfdags = dags;
        this.heuristic = new AlphaOrder(this.setOfdags);
    }

    public Dag union() {
        if (this.alpha == null) {
            this.heuristic.computeAlphaH2();
            this.alpha = this.heuristic.alpha;
        }

        ArrayList<BetaToAlpha> metAs = new ArrayList<>();
        for (Dag i : this.setOfdags) {
            Dag out = new Dag(i);
            metAs.add(new BetaToAlpha(out, this.alpha));
        }

        this.numberOfInsertedEdges = 0;
        ArrayList<Dag> setOfOutputDags = new ArrayList<>();

        for (BetaToAlpha transformDagi : metAs) {
            transformDagi.transform();
            this.numberOfInsertedEdges += transformDagi.getNumberOfInsertedEdges();
            setOfOutputDags.add(transformDagi.G);
        }

        this.union = new Dag(this.alpha);
        for (Node nodei : this.alpha) {
            for (Dag d : setOfOutputDags) {
                List<Node> parent = d.getParents(nodei);
                for (Node pa : parent) {
                    if (!this.union.isParentOf(pa, nodei)) {
                        this.union.addEdge(new Edge(pa, nodei, Endpoint.TAIL, Endpoint.ARROW));
                    }
                }
            }

        }
        return this.union;
    }
}
