package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import consensusBN.Experiments.ExperimentMinCutGES;
import consensusBN.Experiments.RandomBN;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.File;
import java.util.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * One network, one r, one method, one JVM.
 *
 * The in-process harness with a thread pool could not measure the large networks
 * honestly: a task that exceeds its cap is reported as a timeout but its thread
 * is never cancelled, so it keeps burning CPU and every later measurement in the
 * same JVM is slowed by the corpses of the earlier ones. On link (724 variables)
 * that degenerated into a crawl.
 *
 * Running one method per process and capping it from outside with `timeout`
 * makes a hang cost exactly one measurement. Exit code 124 from the shell means
 * the method did not finish, which is the feasibility frontier the paper reports.
 *
 * Prints one CSV line to stdout, or nothing if killed.
 *
 * Usage: ProbeScaling <net> <r> <method> [budget]
 *   methods: pairweights, majMCS, majIdx, mmg, mmgcost, mmgskel, hcempty,
 *            gplus, greedyEa, mcbnc
 */
public class ProbeScaling {

    public static void main(String[] args) throws Exception {
        String net = args[0];
        int r = Integer.parseInt(args[1]);
        String method = args[2];
        int budget = args.length > 3 ? Integer.parseInt(args[3]) : Integer.MAX_VALUE;

        List<Dag> dags = CacheDir.clients(net, r);
        if (dags == null) { System.out.println("cache-too-small"); return; }

        long t0 = System.currentTimeMillis();
        PairWeights w = new PairWeights(dags);
        double wTime = (System.currentTimeMillis() - t0) / 1000.0;

        TreewidthBudget tb = budget == Integer.MAX_VALUE
                ? TreewidthBudget.unbounded() : new TreewidthBudget(budget);

        long t1 = System.currentTimeMillis();
        Dag g;
        switch (method) {
            case "pairweights" -> g = new Dag(w.nodes());
            case "majMCS" -> g = OrientationChooser.majoritySkeletonOriented(w);
            case "majIdx" -> g = w.majoritySkeletonDag();
            case "mmg" -> { MajorityGreedy m = new MajorityGreedy(dags);
                            m.fastCandidateUpdate = true; g = m.search(tb); }
            case "mmgcost" -> { MajorityGreedy m = new MajorityGreedy(dags);
                                m.fastCandidateUpdate = true; m.costAwareRanking = true; g = m.search(tb); }
            case "mmgskel" -> { MajorityGreedy m = new MajorityGreedy(dags);
                                m.useSkeletonObjective = true; g = m.search(tb); }
            case "hcempty" -> g = new StructuralHillClimb(w).search(new Dag(w.nodes()), tb);
            // G+ construction is timed INSIDE the descent, because MCBNC builds it
            // in its own constructor and a comparison that charged only one of the
            // two for it would be meaningless.
            case "hcgplus" -> { Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
                                if (!tb.fits(gp)) { System.out.println("infeasible-gplus"); return; }
                                g = new StructuralHillClimb(w).search(gp, tb); }
            case "gplus" -> g = ConsensusUnion.fusionUnion(dags);
            case "greedyEa" -> g = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(budget));
            case "mcbnc" -> g = new MinCutTreeWidthUnion(dags, 10, budget).fusion();
            // MCBNC as the AAAI paper actually runs it: sweep the criticality
            // threshold all the way down and select the snapshot closest to the
            // inputs. Passing an unbounded treewidth to the plain "mcbnc" arm makes
            // it a no-op that returns G+ untouched, which is not the method.
            case "mcbnctraj" -> {
                MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY);
                mc.experiments_perc = true;
                mc.fusion();
                Dag best = null; double bestVal = Double.POSITIVE_INFINITY;
                for (Dag snap : mc.outputExperimentDAGs) {
                    double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(snap)));
                    if (v < bestVal && tb.fits(snap)) { bestVal = v; best = snap; }
                }
                if (best == null) { System.out.println("no-feasible-snapshot"); return; }
                g = best;
            }
            default -> throw new IllegalArgumentException("unknown method " + method);
        }
        double time = (System.currentTimeMillis() - t1) / 1000.0;

        Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());
        double smhd = w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
        double smhdGold = Utils.SMHD(g, gold);
        int shdGold = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)),
                                     GraphTransforms.dagToCpdag(new EdgeListGraph(gold)));
        double meanInputGold = 0;
        for (Dag d : dags) meanInputGold += Utils.SMHD(d, gold);
        meanInputGold /= r;

        System.out.printf(Locale.US, "%s,%d,%d,%s,%s,%d,%d,%.4f,%.4f,%d,%.4f,%.4f,%.4f%n",
                net, w.n(), r, method, budget == Integer.MAX_VALUE ? "none" : String.valueOf(budget),
                g.getNumEdges(), Utils.getTreeWidth(g), smhd, smhdGold, shdGold, time, wTime, meanInputGold);
    }

}
