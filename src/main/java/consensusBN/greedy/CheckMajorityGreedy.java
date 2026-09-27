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
 * Properties MMG must satisfy:
 *   1. the returned graph is a DAG within the budget,
 *   2. the result never scores worse than the empty graph (the search starts
 *      there and only applies strictly improving moves),
 *   3. it never beats the lower bound of Proposition 3,
 *   4. unbounded and under the skeleton objective it reproduces the optimum
 *      exactly, which is the one case where the true optimum is known.
 *
 * Usage: CheckMajorityGreedy
 */
public class CheckMajorityGreedy {

    public static void main(String[] args) throws Exception {
        boolean ok = true;
        ok &= run("alarm.0", 10, 0);
        ok &= run("insurance.0", 10, 1);
        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    static boolean run(String net, int nDags, int seed) throws Exception {
        String netName = net.split("\\.")[0];
        DataSet data = readData("./res/networks/BBDD/" + netName + "/" + net + ".csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + netName + ".xbif");
        RandomBN rbn = new RandomBN(reader, data, seed, nDags, 5);
        rbn.generate();
        List<Dag> dags = rbn.setOfRandomDags;

        boolean ok = true;
        System.out.printf("== %s r=%d ==%n", net, nDags);

        double emptySmhd = Utils.SMHD(new Dag(dags.get(0).getNodes()), dags);

        for (int budget : new int[]{3, 5, 8, Integer.MAX_VALUE}) {
            MajorityGreedy mmg = new MajorityGreedy(dags);
            TreewidthBudget tb = budget == Integer.MAX_VALUE
                    ? TreewidthBudget.unbounded() : new TreewidthBudget(budget);
            Dag result = mmg.search(tb);

            double smhd = Utils.SMHD(result, dags);
            boolean inBudget = tb.fits(result);
            boolean notWorseThanEmpty = smhd <= emptySmhd + 1e-9;
            boolean aboveBound = smhd >= mmg.weights().smhdLowerBound() - 1e-9;
            ok &= inBudget && notWorseThanEmpty && aboveBound;

            System.out.printf(Locale.US,
                    "  moral  budget=%-6s edges=%4d tw=%3d smhd=%9.4f (empty %.4f, LB %.4f) %s %s %s  %.2fs%n",
                    budget == Integer.MAX_VALUE ? "none" : budget,
                    result.getNumEdges(), Utils.getTreeWidth(result), smhd, emptySmhd,
                    mmg.weights().smhdLowerBound(),
                    inBudget ? "inBudget" : "FAIL-budget",
                    notWorseThanEmpty ? "<=empty" : "FAIL-empty",
                    aboveBound ? ">=LB" : "FAIL-LB", mmg.executionTime);
        }

        // Skeleton objective, unbounded: the optimum is known and must be attained.
        MajorityGreedy skel = new MajorityGreedy(dags);
        skel.useSkeletonObjective = true;
        Dag sResult = skel.search(TreewidthBudget.unbounded());
        double sVal = 0;
        for (Dag g : dags) sVal += Utils.SMHDwithoutMoralize(sResult, g);
        sVal /= dags.size();
        boolean attains = Math.abs(sVal - skel.weights().skeletonOptimum()) < 1e-9;
        ok &= attains;
        System.out.printf(Locale.US, "  skeleton unbounded: %.4f vs optimum %.4f -> %s%n",
                sVal, skel.weights().skeletonOptimum(), attains ? "ok" : "FAIL");

        // Reference point, not an assertion: how the incumbent greedy does.
        Dag ea = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", "5");
        System.out.printf(Locale.US, "  [ref] Greedy-Ea tw5 smhd=%.4f edges=%d tw=%d%n",
                Utils.SMHD(ea, dags), ea.getNumEdges(), Utils.getTreeWidth(ea));
        return ok;
    }
}
