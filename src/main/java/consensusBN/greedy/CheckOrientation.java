package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.util.*;

/**
 * What orientation must satisfy, and what it should buy.
 *
 * Invariants (asserted):
 *   1. re-orienting preserves the skeleton exactly, so the skeleton score is
 *      unchanged — this is the orientation asymmetry made operational,
 *   2. the result is a DAG,
 *   3. it never scores worse on SMHD than the arbitrary index-order orientation
 *      that majoritySkeletonDag currently produces.
 *
 * Measured, not asserted: how much moralisation cost the MCS ordering saves, and
 * whether the skeleton-optimal DAG becomes competitive on SMHD once it is
 * oriented sensibly. That is the interesting question, since the two objectives
 * were shown to pull apart only when orientation was left arbitrary.
 *
 * Usage: CheckOrientation
 */
public class CheckOrientation {

    public static void main(String[] args) {
        boolean ok = true;
        System.out.printf("%-12s %4s %6s %8s %8s %8s %8s %8s%n",
                "net", "r", "edges", "SMHDidx", "SMHDmcs", "moralIdx", "moralMcs", "skelSame");
        for (String net : new String[]{"alarm", "insurance", "child", "water",
                                       "hepar2", "win95pts", "pathfinder"}) {
            for (int r : new int[]{10, 30}) {
                Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(
                        new File("./results/Cache/cachedDAGs-" + net + ".ser"));
                if (cached.size() < r) continue;
                List<Dag> dags = new ArrayList<>();
                for (int i = 0; i < r; i++) dags.add(cached.get(i));
                PairWeights w = new PairWeights(dags);

                Dag idx = w.majoritySkeletonDag();
                Dag mcs = OrientationChooser.majoritySkeletonOriented(w);

                double smhdIdx = w.smhdOf(w.objectiveOfMoral(Utils.moralize(idx)));
                double smhdMcs = w.smhdOf(w.objectiveOfMoral(Utils.moralize(mcs)));
                double skelIdx = w.skelOf(w.objectiveOfSkeleton(idx));
                double skelMcs = w.skelOf(w.objectiveOfSkeleton(mcs));

                boolean sameSkel = Math.abs(skelIdx - skelMcs) < 1e-9
                        && idx.getNumEdges() == mcs.getNumEdges();
                boolean isDag = !mcs.paths().existsDirectedCycle();
                boolean notWorse = smhdMcs <= smhdIdx + 1e-9;
                ok &= sameSkel && isDag && notWorse;

                System.out.printf(Locale.US, "%-12s %4d %6d %8.2f %8.2f %8d %8d %8s%s%s%n",
                        net, r, mcs.getNumEdges(), smhdIdx, smhdMcs,
                        OrientationChooser.moralisationCost(idx),
                        OrientationChooser.moralisationCost(mcs),
                        sameSkel ? "ok" : "FAIL",
                        isDag ? "" : " FAIL-cycle",
                        notWorse ? "" : " FAIL-worse");
            }
        }
        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }
}
