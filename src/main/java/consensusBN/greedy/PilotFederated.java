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
 * The federated pilot: the same comparison as PilotMajorityGreedy but on the
 * GES-learned client DAGs cached in results/Cache, which is the regime the
 * paper actually claims, and on networks large enough to test the scalability
 * argument (link has 724 variables, pigs 441, diabetes 413).
 *
 * Every algorithm runs under a wall-clock cap. A timeout is not a failure of the
 * harness, it is the measurement: which methods remain feasible at which size is
 * the feasibility frontier the paper reports. Timed-out runs are written with
 * status=timeout and empty metrics.
 *
 * Distances to the inputs are computed through PairWeights rather than
 * Utils.SMHD. The two agree exactly (CheckPairWeights) but the closed form is
 * O(|moral(R)|) instead of O(r n^2), which is what makes scoring link tractable
 * at all.
 *
 * Usage: PilotFederated [nets] [rs] [timeoutSeconds]
 *   e.g. PilotFederated alarm,insurance,hepar2,win95pts,pathfinder,andes 10,30,100 300
 */
public class PilotFederated {

    static ExecutorService pool;
    static long timeoutSec;

    public static void main(String[] args) throws Exception {
        String[] nets = (args.length > 0 ? args[0]
                : "alarm,insurance,hepar2,win95pts,pathfinder,andes,pigs,diabetes,link").split(",");
        String[] rs = (args.length > 1 ? args[1] : "10,30,100").split(",");
        timeoutSec = args.length > 2 ? Long.parseLong(args[2]) : 300;
        int[] budgets = {3, 5, 8, 12, Integer.MAX_VALUE};

        pool = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);      // a timed-out algorithm must not keep the JVM alive
            return t;
        });

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/pilot_federated.csv"));
        out.println("net,n,r,budget,algorithm,status,edges,tw,smhd,skel,smhdGold,shdGold,time,"
                + "smhdLowerBound,skeletonOptimum,inputMeanTw,inputMeanSmhdGold");

        for (String net : nets) {
            File cacheFile = new File("./results/Cache/cachedDAGs-" + net + ".ser");
            Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(cacheFile);
            if (cached.isEmpty()) { System.out.println("skip " + net + ": empty cache"); continue; }

            Dag gold = loadGold(net);

            for (String rStr : rs) {
                int r = Integer.parseInt(rStr);
                if (cached.size() < r) continue;

                List<Dag> dags = new ArrayList<>();
                for (int i = 0; i < r; i++) dags.add(cached.get(i));

                long t0 = System.currentTimeMillis();
                PairWeights w = new PairWeights(dags);
                double precomputeSec = (System.currentTimeMillis() - t0) / 1000.0;

                double inputMeanTw = 0, inputMeanSmhdGold = 0;
                for (Dag d : dags) {
                    inputMeanTw += Utils.getTreeWidth(d);
                    inputMeanSmhdGold += Utils.SMHD(d, gold);
                }
                inputMeanTw /= r;
                inputMeanSmhdGold /= r;

                System.out.printf(Locale.US, "%n=== %s n=%d r=%d (PairWeights %.2fs, input tw %.1f) ===%n",
                        net, w.n(), r, precomputeSec, inputMeanTw);

                Ctx c = new Ctx(out, net, r, w, gold, dags, inputMeanTw, inputMeanSmhdGold);

                run(c, "none", "empty", () -> new Dag(w.nodes()));
                run(c, "none", "majoritySkeletonDag", w::majoritySkeletonDag);
                run(c, "none", "G+", () -> ConsensusUnion.fusionUnion(dags));

                for (int budget : budgets) {
                    TreewidthBudget tb = budget == Integer.MAX_VALUE
                            ? TreewidthBudget.unbounded() : new TreewidthBudget(budget);
                    String bLabel = budget == Integer.MAX_VALUE ? "none" : String.valueOf(budget);

                    run(c, bLabel, "MMG-moral", () -> {
                        MajorityGreedy m = new MajorityGreedy(dags);
                        m.fastCandidateUpdate = true;
                        return m.search(tb);
                    });
                    run(c, bLabel, "MMG-skel", () -> {
                        MajorityGreedy m = new MajorityGreedy(dags);
                        m.useSkeletonObjective = true;
                        return m.search(tb);
                    });
                    if (budget != Integer.MAX_VALUE) {
                        run(c, bLabel, "Greedy-Ea",
                                () -> ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(budget)));
                        run(c, bLabel, "MCBNC",
                                () -> new MinCutTreeWidthUnion(dags, 10, budget).fusion());
                    }
                }
                out.flush();
            }
        }
        out.close();
        System.out.println("wrote results/greedy/pilot_federated.csv");
        System.exit(0);   // daemon threads from timed-out algorithms may still be running
    }

    record Ctx(PrintWriter out, String net, int r, PairWeights w, Dag gold,
               List<Dag> dags, double inputMeanTw, double inputMeanSmhdGold) {}

    static void run(Ctx c, String budget, String alg, Callable<Dag> task) {
        long t0 = System.currentTimeMillis();
        Dag g;
        try {
            g = pool.submit(task).get(timeoutSec, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            c.out().printf(Locale.US, "%s,%d,%d,%s,%s,timeout,,,,,,,%.4f,%.4f,%.4f,%.2f,%.4f%n",
                    c.net(), c.w().n(), c.r(), budget, alg, (double) timeoutSec,
                    c.w().smhdLowerBound(), c.w().skeletonOptimum(), c.inputMeanTw(), c.inputMeanSmhdGold());
            System.out.printf("  %-20s budget=%-4s TIMEOUT (>%ds)%n", alg, budget, timeoutSec);
            return;
        } catch (Exception e) {
            c.out().printf(Locale.US, "%s,%d,%d,%s,%s,error,,,,,,,,%.4f,%.4f,%.2f,%.4f%n",
                    c.net(), c.w().n(), c.r(), budget, alg,
                    c.w().smhdLowerBound(), c.w().skeletonOptimum(), c.inputMeanTw(), c.inputMeanSmhdGold());
            System.out.printf("  %-20s budget=%-4s ERROR %s%n", alg, budget, e.getCause());
            return;
        }
        double time = (System.currentTimeMillis() - t0) / 1000.0;

        // Closed form: exact and O(|moral(R)|), see CheckPairWeights.
        double smhd = c.w().smhdOf(c.w().objectiveOfMoral(Utils.moralize(g)));
        double skel = c.w().skelOf(c.w().objectiveOfSkeleton(g));
        double smhdGold = Utils.SMHD(g, c.gold());
        int shdGold = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)),
                                     GraphTransforms.dagToCpdag(new EdgeListGraph(c.gold())));

        c.out().printf(Locale.US, "%s,%d,%d,%s,%s,ok,%d,%d,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.2f,%.4f%n",
                c.net(), c.w().n(), c.r(), budget, alg, g.getNumEdges(), Utils.getTreeWidth(g),
                smhd, skel, smhdGold, shdGold, time,
                c.w().smhdLowerBound(), c.w().skeletonOptimum(), c.inputMeanTw(), c.inputMeanSmhdGold());
        System.out.printf(Locale.US, "  %-20s budget=%-4s edges=%5d tw=%4d smhd=%10.2f gold=%8.2f %8.2fs%n",
                alg, budget, g.getNumEdges(), Utils.getTreeWidth(g), smhd, smhdGold, time);
    }

    static Dag loadGold(String net) throws Exception {
        DataSet seedData = readData("./res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + net + ".xbif");
        RandomBN rbnGold = new RandomBN(reader, seedData, 0, 1, 5);
        return new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());
    }
}
