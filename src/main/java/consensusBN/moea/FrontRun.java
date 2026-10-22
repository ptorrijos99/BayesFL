package consensusBN.moea;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import consensusBN.greedy.*;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * One network, one r, every disjoint replicate: the treewidth / fidelity front
 * of each method, in long format (one row per method and level k).
 *
 * Per-level baselines run once per budget k = kMin..kMax and their fronts are
 * the staircase of their outputs; the memetic search produces the whole front
 * in one run. Levels follow the max-clique-size convention of TreewidthBudget.
 *
 * Usage: FrontRun <cacheDir> <net> <r> <outCsv> <reps> <seconds> <seed> [ga|noga] [kCap]
 */
public class FrontRun {

    static final String HEADER = "sampleSize,net,n,r,rep,seed,kMax,smhdLB,method,k,"
            + "edges,tw,smhdInputs,skelInputs,smhdGold,shdGold,time";

    public static void main(String[] args) throws Exception {
        String dir = args[0], net = args[1];
        int r = Integer.parseInt(args[2]);
        String outPath = args[3];
        int maxReps = Integer.parseInt(args[4]);
        double seconds = Double.parseDouble(args[5]);
        long seed = Long.parseLong(args[6]);
        boolean withGa = args.length > 7 && args[7].equals("ga");
        // "light" skips the forward baselines (MMG, Greedy-E_a), which do not scale to
        // the large networks of the scalability study and are not reported there.
        boolean light = args.length > 7 && args[7].equals("light");
        int kCap = args.length > 8 ? Integer.parseInt(args[8]) : 20;
        // Optional single replicate, so a cluster array can run one replicate per task.
        int onlyRep = args.length > 9 ? Integer.parseInt(args[9]) : -1;
        System.setProperty("cacheDir", dir);
        TreewidthBudget.fast = true;
        String tag = CacheDir.tag();

        Map<Integer, Dag> cached = CacheDir.read(net);
        int reps = Math.min(maxReps, cached.size() / r);
        File outFile = new File(outPath);
        if (outFile.getParentFile() != null) outFile.getParentFile().mkdirs();
        boolean fresh = !outFile.exists();
        PrintWriter out = new PrintWriter(new FileWriter(outFile, true));
        if (fresh) out.println(HEADER + ",shdInputs");
        PrintWriter trace = new PrintWriter(new FileWriter(outPath.replace(".csv", "_trace.csv"), true));

        for (int rep = 0; rep < reps; rep++) {
            if (onlyRep >= 0 && rep != onlyRep) continue;
            List<Dag> dags = new ArrayList<>();
            for (int i = rep * r; i < (rep + 1) * r; i++) dags.add(cached.get(i));

            long t0 = System.currentTimeMillis();
            PairWeights w = new PairWeights(dags);
            double wTime = secs(t0);
            Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());
            Graph goldCpdag = GraphTransforms.dagToCpdag(new EdgeListGraph(gold));
            List<Graph> inCpdags = new ArrayList<>();
            for (Dag d : dags) inCpdags.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
            double lex = 1.0 / (2 * w.r());

            // Top of the front: the unbounded G+-free climbs. Levels run up to
            // the treewidth they reach, where the budget stops binding.
            List<Dag> anchors = new ArrayList<>();
            long ta = System.currentTimeMillis();
            for (Dag start : List.of(new Dag(w.nodes()), OrientationChooser.majoritySkeletonOriented(w))) {
                StructuralHillClimb s = new StructuralHillClimb(w);
                s.skelWeight = lex;
                anchors.add(s.search(start, TreewidthBudget.unbounded()));
            }
            double anchorTime = secs(ta);
            int kMin = 2, kMax = kMin;
            for (Dag a : anchors) kMax = Math.max(kMax, TreewidthBudget.treewidth(a));
            kMax = Math.min(kMax, kCap);

            String prefix = String.format(Locale.US, "%s,%s,%d,%d,%d,%d,%d,%.4f",
                    tag, net, w.n(), r, rep, seed, kMax, w.smhdLowerBound());
            Emit emit = (method, k, g, time) -> {
                out.println(prefix + "," + method + "," + k + "," + metrics(w, g, gold, goldCpdag, time)
                        + String.format(Locale.US, ",%.4f", shdToInputs(g, inCpdags)));
                out.flush();
            };

            // Per-level baselines. Each list holds the output at k, and the
            // front of a method at k is its best output at any level <= k.
            Map<String, Dag[]> outs = new LinkedHashMap<>();
            Map<String, Double> times = new LinkedHashMap<>();
            for (String m : light ? List.of("PHC") : List.of("PHC", "MMG", "Greedy-Ea")) {
                outs.put(m, new Dag[kMax + 1]);
                times.put(m, wTime);
            }
            for (int k = kMin; k <= kMax; k++) {
                TreewidthBudget tb = new TreewidthBudget(k);
                long t = System.currentTimeMillis();
                StructuralHillClimb s = new StructuralHillClimb(w);
                s.skelWeight = lex;
                outs.get("PHC")[k] = s.search(new Dag(w.nodes()), tb);
                times.merge("PHC", secs(t), Double::sum);
                if (light) continue;

                t = System.currentTimeMillis();
                MajorityGreedy mg = new MajorityGreedy(dags);
                mg.fastCandidateUpdate = true;
                outs.get("MMG")[k] = mg.search(tb);
                times.merge("MMG", secs(t), Double::sum);

                t = System.currentTimeMillis();
                outs.get("Greedy-Ea")[k] = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(k));
                times.merge("Greedy-Ea", secs(t), Double::sum);
            }
            if (withGa) {
                outs.put("GA-Ec", new Dag[kMax + 1]);
                times.put("GA-Ec", 0.0);
                ConsensusUnion.metricSMHD = true;
                ConsensusUnion.metricAgainstOriginalDAGs = true;
                for (int k = kMin; k <= kMax; k++) {
                    long t = System.currentTimeMillis();
                    GeneticTreeWidthUnion ga = new GeneticTreeWidthUnion(dags, (int) seed, k);
                    ga.populationSize = 100;
                    ga.numIterations = 100;
                    ga.candidatesFromInitialDAGs = true;
                    ga.repeatCandidates = false;
                    ga.useGreedyWarmstart = true;
                    outs.get("GA-Ec")[k] = ga.fusionUnion();
                    times.merge("GA-Ec", secs(t), Double::sum);
                }
            }

            long tk = System.currentTimeMillis();
            Dag kruskal = JournalRun.kruskalForest(w);
            double kruskalTime = wTime + secs(tk);
            emit.row("Kruskal", 2, kruskal, kruskalTime);

            TreewidthArchiveMA probe = new TreewidthArchiveMA(w, kMin, kMax, seed);
            for (Map.Entry<String, Dag[]> e : outs.entrySet())
                emitFront(emit, probe, e.getKey(), e.getValue(), kMin, kMax, times.get(e.getKey()));

            // Memetic search, seeded with the PHC front, Kruskal and the anchors,
            // and charged for all of them. Ablations: the same archive with the
            // kick as the only operator, and an independent iterated local search
            // per level (kick only, own archive) splitting the same total time.
            double seedTime = times.get("PHC") + kruskalTime - wTime + anchorTime;
            for (String name : List.of("TAMA", "TAMA-kick", "ILS-k")) {
                long t = System.currentTimeMillis();
                Dag[] front = new Dag[kMax + 1];
                int evals = 0;
                StringBuilder ops = new StringBuilder();
                List<TreewidthArchiveMA> runs = new ArrayList<>();
                if (name.equals("ILS-k")) {
                    for (int k = kMin; k <= kMax; k++) {
                        TreewidthArchiveMA ma = new TreewidthArchiveMA(w, k, k, seed + k);
                        ma.ops = EnumSet.of(TreewidthArchiveMA.Op.KICK);
                        ma.timeLimit = seconds / (kMax - kMin + 1);
                        ma.offer(outs.get("PHC")[k]);
                        if (k == kMin) ma.offer(kruskal);
                        ma.run();
                        front[k] = ma.frontAt(k);
                        evals += ma.evaluations;
                    }
                } else {
                    TreewidthArchiveMA ma = new TreewidthArchiveMA(w, kMin, kMax, seed);
                    if (name.equals("TAMA-kick")) ma.ops = EnumSet.of(TreewidthArchiveMA.Op.KICK);
                    ma.timeLimit = seconds;
                    for (int k = kMin; k <= kMax; k++) ma.offer(outs.get("PHC")[k]);
                    ma.offer(kruskal);
                    for (Dag a : anchors) ma.offer(a);
                    ma.run();
                    for (int k = kMin; k <= kMax; k++) front[k] = ma.frontAt(k);
                    evals = ma.evaluations;
                    for (Map.Entry<TreewidthArchiveMA.Op, int[]> e : ma.opStats.entrySet())
                        ops.append(' ').append(e.getKey()).append('=').append(e.getValue()[1]).append('/').append(e.getValue()[0]);
                    for (double[] row : ma.trace) {
                        StringBuilder sb = new StringBuilder();
                        for (int k = kMin; k <= kMax; k++)
                            sb.append(String.format(Locale.US, ",%.4f", w.smhdOf(row[k + 1])));
                        trace.printf(Locale.US, "%s,%s,%d,%d,%d,%s,%d,%.3f%s%n", tag, net, r, rep, seed, name, kMin, row[0], sb);
                    }
                    trace.flush();
                }
                double time = seedTime + secs(t);
                // ILS-k has no archive across levels: its front is the staircase of its levels.
                Dag[] stair = front;
                if (name.equals("ILS-k")) {
                    TreewidthArchiveMA acc = new TreewidthArchiveMA(w, kMin, kMax, 0);
                    for (Dag g : front) if (g != null) acc.offer(g);
                    stair = new Dag[kMax + 1];
                    for (int k = kMin; k <= kMax; k++) stair[k] = acc.frontAt(k);
                }
                for (int k = kMin; k <= kMax; k++) emit.row(name, k, stair[k], time);
                System.out.printf("%s rep %d %s evals=%d%s%n", net, rep, name, evals, ops);
            }
            System.out.printf("%s %s r=%d rep %d/%d kMax=%d done%n", tag, net, r, rep + 1, reps, kMax);
        }
        out.close();
        trace.close();
    }

    interface Emit { void row(String method, int k, Dag g, double time); }

    /**
     * Staircase: at each k, the best of the method's outputs whose MEASURED
     * treewidth is at most k (a baseline run at budget k may land lower, and a
     * method with its own treewidth routine may land elsewhere).
     */
    static void emitFront(Emit emit, TreewidthArchiveMA scorer, String name, Dag[] byK,
                          int kMin, int kMax, double time) {
        TreewidthArchiveMA a = new TreewidthArchiveMA(scorer.weights(), kMin, kMax, 0);
        for (Dag g : byK) if (g != null) a.offer(g);
        for (int k = kMin; k <= kMax; k++) {
            Dag best = a.frontAt(k);
            if (best != null) emit.row(name, k, best, time);
        }
    }

    /** Mean SHD between the CPDAG of g and the CPDAGs of the inputs. */
    static double shdToInputs(Dag g, List<Graph> inCpdags) {
        Graph c = GraphTransforms.dagToCpdag(new EdgeListGraph(g));
        double s = 0;
        for (Graph x : inCpdags) s += Utils.SHDundir(c, x);
        return s / inCpdags.size();
    }

    static double secs(long t0) { return (System.currentTimeMillis() - t0) / 1000.0; }

    static String metrics(PairWeights w, Dag g, Dag gold, Graph goldCpdag, double time) {
        double smhdIn = w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
        double skelIn = w.skelOf(w.objectiveOfSkeleton(g));
        int shd = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)), goldCpdag);
        return String.format(Locale.US, "%d,%d,%.4f,%.4f,%d,%d,%.4f",
                g.getNumEdges(), TreewidthBudget.treewidth(g), smhdIn, skelIn,
                Utils.SMHD(g, gold), shd, time);
    }
}
