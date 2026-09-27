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
 * Verifies PairWeights against the identities validated in DiagDecomposition:
 * the closed form must reproduce Utils.SMHD and the mean skeleton difference
 * exactly, the skeleton optimum must be attained by majoritySkeletonDag(), and
 * the SMHD lower bound must not be violated by any candidate.
 *
 * Usage: CheckPairWeights
 */
public class CheckPairWeights {

    public static void main(String[] args) throws Exception {
        boolean ok = true;
        ok &= check("alarm.0", 10, 0);
        ok &= check("insurance.0", 20, 1);
        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    static boolean check(String net, int nDags, int seed) throws Exception {
        String netName = net.split("\\.")[0];
        DataSet data = readData("./res/networks/BBDD/" + netName + "/" + net + ".csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + netName + ".xbif");
        RandomBN rbn = new RandomBN(reader, data, seed, nDags, 5);
        rbn.generate();
        List<Dag> dags = rbn.setOfRandomDags;

        PairWeights w = new PairWeights(dags);
        Dag gPlus = ConsensusUnion.fusionUnion(dags);

        Map<String, Dag> cands = new LinkedHashMap<>();
        cands.put("empty", new Dag(w.nodes()));
        cands.put("G+", gPlus);
        cands.put("greedyEa-tw5", ConsensusUnion.fusionUnion(dags, "MaxTreewidth", "5"));
        Random rnd = new Random(seed);
        Dag sub = new Dag(gPlus);
        for (Edge e : new ArrayList<>(gPlus.getEdges())) if (rnd.nextDouble() > 0.5) sub.removeEdge(e);
        cands.put("randsub-0.50", sub);

        boolean ok = true;
        System.out.printf("== %s r=%d n=%d ==%n", net, w.r(), w.n());
        for (Map.Entry<String, Dag> c : cands.entrySet()) {
            Dag R = c.getValue();
            double smhdDirect = Utils.SMHD(R, dags);
            double smhdClosed = w.smhdOf(w.objectiveOfMoral(Utils.moralize(R)));
            double skelDirect = meanSkeletonDiff(R, dags);
            double skelClosed = w.skelOf(w.objectiveOfSkeleton(R));

            boolean a = Math.abs(smhdDirect - smhdClosed) < 1e-9;
            boolean b = Math.abs(skelDirect - skelClosed) < 1e-9;
            boolean d = smhdDirect >= w.smhdLowerBound() - 1e-9;
            ok &= a && b && d;
            System.out.printf(Locale.US, "  %-14s smhd %.4f/%.4f %s  skel %.4f/%.4f %s  >=LB %s%n",
                    c.getKey(), smhdDirect, smhdClosed, a ? "ok" : "FAIL",
                    skelDirect, skelClosed, b ? "ok" : "FAIL", d ? "ok" : "FAIL");
        }

        Dag maj = w.majoritySkeletonDag();
        boolean e = Math.abs(meanSkeletonDiff(maj, dags) - w.skeletonOptimum()) < 1e-9;
        ok &= e;
        System.out.printf(Locale.US, "  majority-skeleton attains optimum: %s (%.4f vs %.4f)%n",
                e ? "ok" : "FAIL", meanSkeletonDiff(maj, dags), w.skeletonOptimum());

        boolean f = !w.supportedPairs().isEmpty();
        ok &= f;
        System.out.printf("  supportedPairs non-empty: %s (%d)%n", f ? "ok" : "FAIL", w.supportedPairs().size());
        return ok;
    }

    static double meanSkeletonDiff(Dag R, List<Dag> dags) {
        double s = 0;
        for (Dag g : dags) s += Utils.SMHDwithoutMoralize(R, g);
        return s / dags.size();
    }
}
