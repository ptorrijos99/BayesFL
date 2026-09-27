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
 * The budget must agree, edge case for edge case, with the acceptance rule
 * already used by ConsensusUnion.applyGreedyMaxTreewidth, which rejects when
 * some clique has size > maxCliqueSize. Any disagreement is an off-by-one that
 * would silently break comparability with the existing parquet results.
 *
 * Usage: CheckTreewidthBudget
 */
public class CheckTreewidthBudget {

    public static void main(String[] args) throws Exception {
        String net = "alarm.0";
        DataSet data = readData("./res/networks/BBDD/alarm/" + net + ".csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/alarm.xbif");
        RandomBN rbn = new RandomBN(reader, data, 0, 10, 5);
        rbn.generate();
        List<Dag> dags = rbn.setOfRandomDags;

        Dag gPlus = ConsensusUnion.fusionUnion(dags);
        List<Dag> cands = new ArrayList<>();
        cands.add(new Dag(gPlus.getNodes()));
        cands.add(gPlus);
        Random rnd = new Random(0);
        for (int k = 0; k < 8; k++) {
            Dag sub = new Dag(gPlus);
            for (Edge e : new ArrayList<>(gPlus.getEdges()))
                if (rnd.nextDouble() > 0.2 + 0.1 * k) sub.removeEdge(e);
            cands.add(sub);
        }

        boolean ok = true;
        for (int budget : new int[]{1, 2, 3, 5, 8, 13, 30}) {
            TreewidthBudget tb = new TreewidthBudget(budget);
            for (Dag d : cands) {
                boolean mine = tb.fits(d);
                boolean reference = true;   // the rule inside applyGreedyMaxTreewidth
                for (Set<Node> clique : Utils.getMoralTriangulatedCliques(d).values())
                    if (clique.size() > budget) { reference = false; break; }
                if (mine != reference) {
                    ok = false;
                    System.out.printf("  FAIL budget=%d edges=%d mine=%s reference=%s tw=%d%n",
                            budget, d.getNumEdges(), mine, reference, Utils.getTreeWidth(d));
                }
            }
        }

        TreewidthBudget un = TreewidthBudget.unbounded();
        if (!un.fits(gPlus)) { ok = false; System.out.println("  FAIL unbounded rejected G+"); }

        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }
}
