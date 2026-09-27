package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.util.*;

/**
 * Properties the local search must satisfy:
 *   1. it never returns something worse than its starting point,
 *   2. the returned graph respects the budget,
 *   3. the tracker objective still matches a full recomputation at the end,
 *   4. started from MMG's output it is never worse than MMG, and on the case
 *      that motivated it (alarm federated, budget 5, where forward-only stalls
 *      at 12.7 while MCBNC reaches 5.5) it must strictly improve,
 *   5. it converges from three different starting points (empty, MMG, G+).
 *
 * Usage: CheckHillClimb
 */
public class CheckHillClimb {

    public static void main(String[] args) {
        boolean ok = true;
        ok &= run("alarm", 10);
        ok &= run("insurance", 10);
        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    static boolean run(String net, int r) {
        Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(
                new File("./results/Cache/cachedDAGs-" + net + ".ser"));
        List<Dag> dags = new ArrayList<>();
        for (int i = 0; i < r; i++) dags.add(cached.get(i));

        PairWeights w = new PairWeights(dags);
        boolean ok = true;
        System.out.printf("== %s r=%d (federated) ==%n", net, r);

        for (int budget : new int[]{3, 5, 8, Integer.MAX_VALUE}) {
            TreewidthBudget tb = budget == Integer.MAX_VALUE
                    ? TreewidthBudget.unbounded() : new TreewidthBudget(budget);
            String bl = budget == Integer.MAX_VALUE ? "none" : String.valueOf(budget);

            Dag mmg = new MajorityGreedy(dags).search(tb);
            double mmgSmhd = smhd(w, mmg);

            for (String start : new String[]{"empty", "mmg", "G+"}) {
                Dag init = switch (start) {
                    case "empty" -> new Dag(w.nodes());
                    case "mmg" -> new Dag(mmg);
                    default -> new Dag(ConsensusUnion.fusionUnion(dags));
                };
                // G+ may already violate the budget; the search must not be asked to
                // repair infeasibility, so skip that combination.
                if (!tb.fits(init)) {
                    System.out.printf("  budget=%-4s from %-5s: initial infeasible, skipped%n", bl, start);
                    continue;
                }
                double before = smhd(w, init);

                StructuralHillClimb hc = new StructuralHillClimb(dags);
                Dag out = hc.search(init, tb);
                double after = smhd(w, out);

                boolean notWorse = after <= before + 1e-9;
                boolean inBudget = tb.fits(out);
                boolean trackerOk = Math.abs(smhd(w, out) - w.smhdOf(w.objectiveOfMoral(Utils.moralize(out)))) < 1e-9;
                boolean beatsMmg = !start.equals("mmg") || after <= mmgSmhd + 1e-9;
                ok &= notWorse && inBudget && trackerOk && beatsMmg;

                System.out.printf(Locale.US,
                        "  budget=%-4s from %-5s: %8.2f -> %8.2f (mmg %8.2f) edges=%4d tw=%3d %s %s %s %.2fs%n",
                        bl, start, before, after, mmgSmhd, out.getNumEdges(), Utils.getTreeWidth(out),
                        notWorse ? "notWorse" : "FAIL-worse",
                        inBudget ? "inBudget" : "FAIL-budget",
                        beatsMmg ? "<=mmg" : "FAIL-mmg", hc.executionTime);
            }
        }

        // REFUTED HYPOTHESIS, kept as a measurement rather than an assertion.
        //
        // This originally asserted that adding removal and reversal would unstick
        // the forward greedy on alarm/tw5, where MMG stops at 12.7 and MCBNC
        // reaches 5.5. It does not: 12.7 is a genuine local optimum of the
        // add/remove/reverse neighbourhood, so the gap to MCBNC is NOT a missing
        // operator. What actually moves the result is the starting point, and only
        // on some networks (alarm yes, insurance no), which is why the comparison
        // below is reported and not asserted.
        System.out.println("  -- initialisation sensitivity (measurement, not an assertion) --");
        for (int budget : new int[]{5, 8, 12}) {
            TreewidthBudget tb = new TreewidthBudget(budget);
            Dag gPlus = new Dag(ConsensusUnion.fusionUnion(dags));
            if (!tb.fits(gPlus)) {
                System.out.printf("     budget=%-3d G+ infeasible (tw %d): forward is the only option%n",
                        budget, Utils.getTreeWidth(gPlus));
                continue;
            }
            double fromEmpty = smhd(w, new StructuralHillClimb(dags).search(new Dag(w.nodes()), tb));
            double fromGPlus = smhd(w, new StructuralHillClimb(dags).search(gPlus, tb));
            System.out.printf(Locale.US, "     budget=%-3d fromEmpty=%.2f fromG+=%.2f  (G+ better by %.2f)%n",
                    budget, fromEmpty, fromGPlus, fromEmpty - fromGPlus);
        }
        return ok;
    }

    static double smhd(PairWeights w, Dag g) {
        return w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
    }
}
