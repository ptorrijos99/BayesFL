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
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * The definitive budget sweep, with the grid anchored on the data instead of
 * fixed.
 *
 * Why this harness exists: the earlier pilots used a fixed grid {3,5,8,12,16},
 * but tw(G+) ranges from 3 (child) to 68 (win95pts, pathfinder). On the hard
 * networks no budget in that grid ever admitted G+, so every cell there was
 * forward-only and the "loose budget" half of the comparison was never actually
 * sampled. Conclusions about descending from G+ were drawn from the easy
 * networks alone.
 *
 * The grid here is per (net, r): the tractable anchors {3,5,8} plus quartiles of
 * tw(G+) up to tw(G+) itself, where G+ is feasible by construction. That spans
 * the whole range from "inference is cheap" to "no constraint at all", and makes
 * budgetFraction = budget / tw(G+) comparable across networks.
 *
 * Every algorithm runs under a wall-clock cap; a timeout is a measurement, not a
 * harness failure.
 *
 * Usage: PilotBudgetSweep [nets] [rs] [timeoutSeconds]
 */
public class PilotBudgetSweep {

    static ExecutorService pool;
    static long timeoutSec;

    public static void main(String[] args) throws Exception {
        String[] nets = (args.length > 0 ? args[0]
                : "alarm,insurance,child,water,mildew,barley,hailfinder,hepar2,win95pts,pathfinder").split(",");
        String[] rs = (args.length > 1 ? args[1] : "10,30,100").split(",");
        timeoutSec = args.length > 2 ? Long.parseLong(args[2]) : 200;

        pool = Executors.newCachedThreadPool(t -> { Thread th = new Thread(t); th.setDaemon(true); return th; });

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/pilot_budget_sweep.csv"));
        out.println("net,n,r,budget,budgetFraction,twGPlus,feasibleGPlus,algorithm,status,"
                + "edges,tw,smhd,skel,smhdGold,shdGold,time,smhdLowerBound,skeletonOptimum,inputMeanTw");

        for (String net : nets) {
            Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(
                    new File("./results/Cache/cachedDAGs-" + net + ".ser"));
            if (cached.isEmpty()) { System.out.println("skip " + net); continue; }
            Dag gold = loadGold(net);

            for (String rStr : rs) {
                int r = Integer.parseInt(rStr);
                if (cached.size() < r) continue;
                List<Dag> dags = new ArrayList<>();
                for (int i = 0; i < r; i++) dags.add(cached.get(i));

                PairWeights w = new PairWeights(dags);
                Dag gPlus = new Dag(ConsensusUnion.fusionUnion(dags));
                int twGPlus = Utils.getTreeWidth(gPlus);

                double inputMeanTw = 0;
                for (Dag d : dags) inputMeanTw += Utils.getTreeWidth(d);
                inputMeanTw /= r;

                // Tractable anchors plus quartiles of tw(G+), so the sweep always
                // reaches the unconstrained end however large tw(G+) happens to be.
                TreeSet<Integer> grid = new TreeSet<>(List.of(3, 5, 8));
                for (double f : new double[]{0.25, 0.5, 0.75, 1.0})
                    grid.add(Math.max(2, (int) Math.ceil(f * twGPlus)));
                grid.removeIf(b -> b > twGPlus);
                grid.add(twGPlus);

                System.out.printf(Locale.US, "%n=== %s n=%d r=%d tw(G+)=%d inputTw=%.1f grid=%s ===%n",
                        net, w.n(), r, twGPlus, inputMeanTw, grid);

                for (int budget : grid) {
                    TreewidthBudget tb = new TreewidthBudget(budget);
                    boolean feas = tb.fits(gPlus);
                    Ctx c = new Ctx(out, net, r, budget, twGPlus, feas, w, gold, inputMeanTw);

                    run(c, "MMG", () -> { MajorityGreedy m = new MajorityGreedy(dags);
                                          m.fastCandidateUpdate = true; return m.search(tb); });
                    run(c, "MMG-cost", () -> { MajorityGreedy m = new MajorityGreedy(dags);
                                               m.fastCandidateUpdate = true;
                                               m.costAwareRanking = true; return m.search(tb); });
                    run(c, "MMG-skel", () -> { MajorityGreedy m = new MajorityGreedy(dags);
                                               m.useSkeletonObjective = true; return m.search(tb); });
                    run(c, "majMCS", () -> OrientationChooser.majoritySkeletonOriented(w));
                    run(c, "HC-empty", () -> new StructuralHillClimb(w).search(new Dag(w.nodes()), tb));
                    run(c, "HC-mmg", () -> {
                        MajorityGreedy m = new MajorityGreedy(dags);
                        m.fastCandidateUpdate = true;
                        return new StructuralHillClimb(w).search(new Dag(m.search(tb)), tb);
                    });
                    if (feas) run(c, "HC-gplus", () -> new StructuralHillClimb(w).search(new Dag(gPlus), tb));
                    run(c, "Greedy-Ea",
                            () -> ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(budget)));
                    run(c, "MCBNC", () -> new MinCutTreeWidthUnion(dags, 10, budget).fusion());
                }
                out.flush();
            }
        }
        out.close();
        System.out.println("wrote results/greedy/pilot_budget_sweep.csv");
        System.exit(0);
    }

    record Ctx(PrintWriter out, String net, int r, int budget, int twGPlus, boolean feas,
               PairWeights w, Dag gold, double inputMeanTw) {}

    static void run(Ctx c, String alg, Callable<Dag> task) {
        long t0 = System.currentTimeMillis();
        Dag g;
        try {
            g = pool.submit(task).get(timeoutSec, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            emitStatus(c, alg, "timeout");
            System.out.printf("  budget=%-3d %-10s TIMEOUT (>%ds)%n", c.budget(), alg, timeoutSec);
            return;
        } catch (Exception e) {
            emitStatus(c, alg, "error");
            System.out.printf("  budget=%-3d %-10s ERROR %s%n", c.budget(), alg, e.getCause());
            return;
        }
        double time = (System.currentTimeMillis() - t0) / 1000.0;

        double smhd = c.w().smhdOf(c.w().objectiveOfMoral(Utils.moralize(g)));
        double skel = c.w().skelOf(c.w().objectiveOfSkeleton(g));
        double smhdGold = Utils.SMHD(g, c.gold());
        int shdGold = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)),
                                     GraphTransforms.dagToCpdag(new EdgeListGraph(c.gold())));

        c.out().printf(Locale.US, "%s,%d,%d,%d,%.4f,%d,%b,%s,ok,%d,%d,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.2f%n",
                c.net(), c.w().n(), c.r(), c.budget(), c.budget() / (double) c.twGPlus(), c.twGPlus(), c.feas(),
                alg, g.getNumEdges(), Utils.getTreeWidth(g), smhd, skel, smhdGold, shdGold, time,
                c.w().smhdLowerBound(), c.w().skeletonOptimum(), c.inputMeanTw());
        System.out.printf(Locale.US, "  budget=%-3d %-10s edges=%5d tw=%3d smhd=%9.2f gold=%8.2f %7.2fs%n",
                c.budget(), alg, g.getNumEdges(), Utils.getTreeWidth(g), smhd, smhdGold, time);
    }

    static void emitStatus(Ctx c, String alg, String status) {
        c.out().printf(Locale.US, "%s,%d,%d,%d,%.4f,%d,%b,%s,%s,,,,,,,%.4f,%.4f,%.4f,%.2f%n",
                c.net(), c.w().n(), c.r(), c.budget(), c.budget() / (double) c.twGPlus(), c.twGPlus(), c.feas(),
                alg, status, (double) timeoutSec, c.w().smhdLowerBound(), c.w().skeletonOptimum(), c.inputMeanTw());
    }

    static Dag loadGold(String net) throws Exception {
        DataSet seedData = readData("./res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + net + ".xbif");
        RandomBN rbnGold = new RandomBN(reader, seedData, 0, 1, 5);
        return new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());
    }
}
