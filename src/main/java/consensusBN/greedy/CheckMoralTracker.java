package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import consensusBN.Experiments.RandomBN;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.util.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * Verifies that the tracker stays exactly in step with a full recomputation
 * through a random walk of insertions AND deletions, and that its predicted
 * gains equal the realised deltas.
 *
 * The deletion case is the one a naive implementation gets wrong: removing an
 * arc x->y does NOT remove the moral edge {x,p} when x and p remain co-parents
 * of some other child. The walk is long enough to hit that case; the harness
 * asserts it was hit at least once, so a vacuous pass is impossible.
 *
 * Usage: CheckMoralTracker
 */
public class CheckMoralTracker {

    public static void main(String[] args) throws Exception {
        boolean ok = true;
        ok &= walk("alarm.0", 10, 0, 400);
        ok &= walk("insurance.0", 10, 1, 400);
        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    static boolean walk(String net, int nDags, int seed, int steps) throws Exception {
        String netName = net.split("\\.")[0];
        DataSet data = readData("./res/networks/BBDD/" + netName + "/" + net + ".csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + netName + ".xbif");
        RandomBN rbn = new RandomBN(reader, data, seed, nDags, 5);
        rbn.generate();
        List<Dag> dags = rbn.setOfRandomDags;

        PairWeights w = new PairWeights(dags);
        Dag gPlus = ConsensusUnion.fusionUnion(dags);
        List<Edge> pool = new ArrayList<>(gPlus.getEdges());

        Dag r = new Dag(w.nodes());
        MoralGraphTracker t = new MoralGraphTracker(w, r);
        Random rnd = new Random(seed);

        int objMismatch = 0, gainMismatch = 0, survivingEdgeCases = 0, adds = 0, removes = 0;

        for (int s = 0; s < steps; s++) {
            boolean doAdd = r.getNumEdges() == 0 || rnd.nextDouble() < 0.65;
            if (doAdd) {
                Edge e = pool.get(rnd.nextInt(pool.size()));
                Node x = e.getNode1(), y = e.getNode2();
                if (rnd.nextBoolean()) { Node tmp = x; x = y; y = tmp; }
                if (r.isAdjacentTo(x, y)) continue;
                if (r.paths().existsDirectedPath(y, x)) continue;

                double before = t.objective();
                double predicted = t.gainAddArc(x, y);
                t.addArc(x, y);
                double after = t.objective();
                if (Math.abs((after - before) - predicted) > 1e-9) gainMismatch++;
                adds++;
            } else {
                List<Edge> present = new ArrayList<>(r.getEdges());
                Edge e = present.get(rnd.nextInt(present.size()));
                Node x = e.getNode1(), y = e.getNode2();
                if (!r.isParentOf(x, y)) { Node tmp = x; x = y; y = tmp; }

                // does {x,y} survive as a moral edge after the arc goes away?
                boolean coParentElsewhere = false;
                for (Node z : r.getChildren(x))
                    if (!z.equals(y) && r.isParentOf(y, z)) coParentElsewhere = true;
                if (coParentElsewhere) survivingEdgeCases++;

                double before = t.objective();
                double predicted = t.gainRemoveArc(x, y);
                t.removeArc(x, y);
                double after = t.objective();
                if (Math.abs((after - before) - predicted) > 1e-9) gainMismatch++;
                removes++;
            }

            double recomputed = w.objectiveOfMoral(Utils.moralize(r));
            if (Math.abs(t.objective() - recomputed) > 1e-9) objMismatch++;
        }

        boolean ok = objMismatch == 0 && gainMismatch == 0 && survivingEdgeCases > 0;
        System.out.printf("%s: adds=%d removes=%d objMismatch=%d gainMismatch=%d survivingCoParentCases=%d -> %s%n",
                net, adds, removes, objMismatch, gainMismatch, survivingEdgeCases, ok ? "ok" : "FAIL");
        return ok;
    }
}
