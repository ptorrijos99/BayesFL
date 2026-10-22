package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * The experiment behind the journal tables: one network, one sample size, one r,
 * one group of methods, every disjoint replicate, long format.
 *
 * Earlier probes each wrote their own wide CSV with their own subset of methods
 * and metrics, which is how the master table ended up without baselines and the
 * variance runs without SHD. This harness writes every method with every metric
 * in one schema, so the paper's tables are a group-by and nothing is stitched.
 *
 * Replicates are DISJOINT client subsets (clients rep*r .. rep*r+r-1), i.e.
 * independent draws of the federation rather than resamples of one.
 *
 * Methods are split into groups so the slow baselines can be capped externally
 * with `timeout` without losing the fast rows of the same cell:
 *   fast     client-mean, medoid, G+, majArb, majMCS, MMG, HC-empty, HC-gplus, best-of-two
 *   mcbnc    MCBNC under its AAAI protocol (theta sweep, selected by distance to the inputs)
 *   mcfix    MCBNC rerun after the snapshot fix, emitting both MCBNC and MCBNC@shd from one sweep
 *   mcfirst  only the point the old sweep never recorded (first point of the theta trajectory)
 *   greedyEa Greedy-Ea under the analogous protocol (treewidth sweep, selected by the inputs)
 *   forward  the G+-free subset of `fast`, for networks where G+ is unaffordable
 *
 * Every selection among candidates uses distance to the INPUTS only; the gold
 * standard is read after the fact, for evaluation.
 *
 * Usage: JournalRun <cacheDir> <net> <r> <group> <outCsv> [maxReps]
 */
public class JournalRun {

    static final String HEADER = "sampleSize,net,source,n,r,rep,noiseSkel,smhdLB,skelOpt,method,param,"
            + "edges,tw,smhdInputs,skelInputs,smhdGold,skelGold,shdGold,time";

    public static void main(String[] args) throws Exception {
        String dir = args[0], net = args[1];
        int r = Integer.parseInt(args[2]);
        String group = args[3], outPath = args[4];
        int maxReps = args.length > 5 ? Integer.parseInt(args[5]) : Integer.MAX_VALUE;
        System.setProperty("cacheDir", dir);
        String tag = CacheDir.tag();

        Map<Integer, Dag> cached = CacheDir.read(net);
        int reps = Math.min(maxReps, cached.size() / r);
        if (reps == 0) { System.out.println("cache-too-small " + net + " " + cached.size()); return; }

        File outFile = new File(outPath);
        outFile.getParentFile().mkdirs();
        boolean fresh = !outFile.exists();
        PrintWriter out = new PrintWriter(new FileWriter(outFile, true));
        if (fresh) out.println(HEADER);

        for (int rep = 0; rep < reps; rep++) {
            List<Dag> dags = new ArrayList<>();
            for (int i = rep * r; i < (rep + 1) * r; i++) {
                Dag d = cached.get(i);
                if (d == null) break;
                dags.add(d);
            }
            if (dags.size() < r) break;

            long tw0 = System.currentTimeMillis();
            PairWeights w = new PairWeights(dags);
            double wTime = (System.currentTimeMillis() - tw0) / 1000.0;
            Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());
            Graph goldCpdag = GraphTransforms.dagToCpdag(new EdgeListGraph(gold));

            int supported = w.supportedPairs().size(), maj = 0;
            for (int i = 0; i < w.n(); i++)
                for (int j = i + 1; j < w.n(); j++)
                    if (2 * w.countSkel(i, j) > r) maj++;
            double noise = supported == 0 ? 0 : 1.0 - maj / (double) supported;

            String prefix = String.format(Locale.US, "%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f",
                    tag, net, net.startsWith("bnrep_") ? "bnRep" : "bnlearn", w.n(), r, rep,
                    noise, w.smhdLowerBound(), w.skeletonOptimum());
            Row row = (method, param, g, time) -> {
                out.println(prefix + "," + method + "," + param + "," + metrics(w, g, gold, goldCpdag, time));
                out.flush();
            };

            switch (group) {
                case "fast", "forward" -> runFast(dags, w, gold, goldCpdag, prefix, out, row, wTime,
                        group.equals("fast"));
                case "variants" -> runVariants(dags, w, row, wTime);
                case "variants2" -> runVariants2(dags, w, row, wTime);
                case "budget" -> runBudget(dags, w, row, wTime);
                case "classvariants" -> runClassVariants(dags, w, row, wTime);
                case "tw1" -> runTw1(dags, w, row, wTime);
                case "shdsel" -> runShdSelection(dags, w, row, wTime);
                case "traj" -> runTrajectory(dags, w, row, wTime);
                case "mcbnc" -> runMcbnc(dags, w, row);
                case "mcfix" -> runMcbncBoth(dags, w, row);
                case "mcfirst" -> runMcbncFirst(dags, row);
                case "greedyEa" -> runGreedyEa(dags, w, row);
                case "mcbudget" -> runMcbncBudget(dags, w, row);
                case "gplus" -> runUnion(dags, w, row, net, r, rep, tag);
                case "phcg" -> runPhcFromUnion(dags, w, row, wTime, net, r, rep, tag);
                case "mcg" -> runMcbncFromUnion(dags, w, row, net, r, rep, tag);
                case "revision" -> runRevision(dags, w, gold, goldCpdag, row, wTime);
                case "avgstart" -> runAvgStart(dags, w, row, wTime);
                case "avgnet" -> {
                    AveragedNetwork an = new AveragedNetwork();
                    long t = System.currentTimeMillis();
                    Dag avg = an.build(w);
                    row.emit("avgNet", String.format(Locale.US, "%.2f", an.threshold), avg,
                            wTime + (System.currentTimeMillis() - t) / 1000.0);
                }
                case "revtraj" -> runRevisionTrajectory(dags, w, row, wTime);
                case "budget2" -> runBudgetSel(dags, w, row, wTime, false);
                case "budget2p" -> runBudgetSel(dags, w, row, wTime, true);
                case "central" -> runCentral(w, row, net, r, rep, Integer.parseInt(tag));
                default -> throw new IllegalArgumentException("unknown group " + group);
            }
            System.out.printf("%s %s r=%d rep %d/%d %s done%n", tag, net, r, rep + 1, reps, group);
        }
        out.close();
    }

    interface Row { void emit(String method, String param, Dag g, double time); }

    static String metrics(PairWeights w, Dag g, Dag gold, Graph goldCpdag, double time) {
        double smhdIn = w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
        double skelIn = w.skelOf(w.objectiveOfSkeleton(g));
        int shd = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)), goldCpdag);
        return String.format(Locale.US, "%d,%d,%.4f,%.4f,%d,%d,%d,%.4f",
                g.getNumEdges(), Utils.getTreeWidth(g), smhdIn, skelIn,
                Utils.SMHD(g, gold), Utils.SMHDwithoutMoralize(g, gold), shd, time);
    }

    static void runFast(List<Dag> dags, PairWeights w, Dag gold, Graph goldCpdag, String prefix,
                        PrintWriter out, Row row, double wTime, boolean withGplus) {
        // The average client, metric by metric: the reference every consensus has
        // to beat to be worth computing.
        double[] acc = new double[8];
        Dag medoid = null; double medVal = Double.POSITIVE_INFINITY;
        for (Dag d : dags) {
            String[] m = metrics(w, d, gold, goldCpdag, 0).split(",");
            for (int k = 0; k < 8; k++) acc[k] += Double.parseDouble(m[k]);
            double v = Double.parseDouble(m[2]);
            if (v < medVal) { medVal = v; medoid = d; }
        }
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < 8; k++) sb.append(k == 0 ? "" : ",").append(String.format(Locale.US, "%.4f", acc[k] / dags.size()));
        out.println(prefix + ",clientMean,-," + sb);
        out.flush();

        // Medoid: the client closest to the others. The data-free way to "just pick
        // one client", which is the obvious alternative to fusing at all.
        long t = System.currentTimeMillis();
        row.emit("medoid", "-", medoid, wTime + (System.currentTimeMillis() - t) / 1000.0);

        t = System.currentTimeMillis();
        Dag arb = w.majoritySkeletonDag();
        row.emit("majArb", "-", arb, wTime + (System.currentTimeMillis() - t) / 1000.0);

        t = System.currentTimeMillis();
        Dag mcs = OrientationChooser.majoritySkeletonOriented(w);
        row.emit("majMCS", "-", mcs, wTime + (System.currentTimeMillis() - t) / 1000.0);

        MajorityGreedy m = new MajorityGreedy(dags);
        m.fastCandidateUpdate = true;
        t = System.currentTimeMillis();
        Dag mmg = m.search(TreewidthBudget.unbounded());
        row.emit("MMG", "-", mmg, wTime + (System.currentTimeMillis() - t) / 1000.0);

        // The hill climbs use the moral objective with the skeleton as a pure
        // tie-breaker (skelWeight < 1/r). The "-moral" rows are the ablation
        // without it, which keeps every arc between co-parents it starts with.
        double lex = 1.0 / (2 * w.r());
        Map<String, Dag> hc = new LinkedHashMap<>();
        Map<String, Double> hcTime = new LinkedHashMap<>();

        hcTime.put("HC-empty", climb(w, new Dag(w.nodes()), lex, row, "HC-empty", wTime, hc));
        climb(w, new Dag(w.nodes()), 0.0, row, "HC-empty-moral", wTime, null);
        hcTime.put("HC-maj", climb(w, OrientationChooser.majoritySkeletonOriented(w), lex, row, "HC-maj", wTime, hc));
        emitBest(w, row, "HC-best-free", hc, hcTime, wTime);

        // Ablation of the orientation tie-break: equal-gain moves in their arbitrary order.
        Map<String, Dag> hcNoDir = new LinkedHashMap<>();
        Map<String, Double> hcNoDirTime = new LinkedHashMap<>();
        hcNoDirTime.put("HC-empty", climb(w, new Dag(w.nodes()), lex, row, "HC-empty-nodir", wTime, hcNoDir, false));
        hcNoDirTime.put("HC-maj", climb(w, OrientationChooser.majoritySkeletonOriented(w), lex, row, "HC-maj-nodir", wTime, hcNoDir, false));
        emitBest(w, row, "HC-best-free-nodir", hcNoDir, hcNoDirTime, wTime);

        if (!withGplus) return;

        t = System.currentTimeMillis();
        Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
        double tGp = (System.currentTimeMillis() - t) / 1000.0;
        row.emit("G+", "-", gp, tGp);

        hcTime.put("HC-gplus", climb(w, new Dag(gp), lex, row, "HC-gplus", wTime + tGp, hc));
        climb(w, new Dag(gp), 0.0, row, "HC-gplus-moral", wTime + tGp, null);
        emitBest(w, row, "HC-best", hc, hcTime, wTime);
    }

    /** Hill-climbs from start, emits the row, returns the time charged. */
    static double climb(PairWeights w, Dag start, double skelWeight, Row row, String name,
                        double overhead, Map<String, Dag> keep) {
        return climb(w, start, skelWeight, row, name, overhead, keep, true);
    }

    static double climb(PairWeights w, Dag start, double skelWeight, Row row, String name,
                        double overhead, Map<String, Dag> keep, boolean dirTieBreak) {
        StructuralHillClimb s = new StructuralHillClimb(w);
        s.skelWeight = skelWeight;
        s.dirTieBreak = dirTieBreak;
        long t = System.currentTimeMillis();
        Dag g = s.search(start, TreewidthBudget.unbounded());
        double time = overhead + (System.currentTimeMillis() - t) / 1000.0;
        row.emit(name, "-", g, time);
        if (keep != null) keep.put(name, g);
        return time;
    }

    /**
     * Runs every start and keeps the output closest to the inputs, the same
     * data-free rule MCBNC uses for theta. Charged for all the starts it ran;
     * the shared pair-weight time is counted once.
     */
    static void emitBest(PairWeights w, Row row, String name, Map<String, Dag> hc, Map<String, Double> times,
                         double wTime) {
        String bestK = null; double bestV = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, Dag> e : hc.entrySet()) {
            double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(e.getValue())))
                    + w.skelOf(w.objectiveOfSkeleton(e.getValue())) * 1e-6;
            if (v < bestV) { bestV = v; bestK = e.getKey(); }
        }
        double total = -wTime * (times.size() - 1);
        for (double x : times.values()) total += x;
        row.emit(name, bestK, hc.get(bestK), total);
    }

    /** Objective ablation: moral only, moral with a skeleton tie-break, and their sum. */
    static void runVariants(List<Dag> dags, PairWeights w, Row row, double wTime) {
        Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
        double[] lambdas = {0.0, 1.0 / (2 * w.r()), 0.5, 1.0};
        String[] names = {"moral", "lex", "half", "sum"};
        for (int k = 0; k < lambdas.length; k++) {
            for (String start : new String[]{"empty", "gplus"}) {
                StructuralHillClimb hc = new StructuralHillClimb(w);
                hc.skelWeight = lambdas[k];
                long t = System.currentTimeMillis();
                Dag g = hc.search(start.equals("empty") ? new Dag(w.nodes()) : new Dag(gp), TreewidthBudget.unbounded());
                row.emit("HC-" + start + "-" + names[k], "-", g, wTime + (System.currentTimeMillis() - t) / 1000.0);
            }
        }
        // Skeleton fixed at the majority graph, orientation searched on the moral objective.
        StructuralHillClimb hc = new StructuralHillClimb(w);
        hc.fixedSkeleton = true;
        long t = System.currentTimeMillis();
        Dag g = hc.search(OrientationChooser.majoritySkeletonOriented(w), TreewidthBudget.unbounded());
        row.emit("majOrient", "-", g, wTime + (System.currentTimeMillis() - t) / 1000.0);
    }

    /**
     * A third starting point that needs no G+: the majority skeleton, oriented
     * by MCS. It is the exact skeleton optimum, costs O(r n^2), and sits much
     * closer to a good consensus than the empty graph does.
     */
    static void runVariants2(List<Dag> dags, PairWeights w, Row row, double wTime) {
        double[] lambdas = {0.0, 1.0 / (2 * w.r()), 0.5, 1.0};
        String[] names = {"moral", "lex", "half", "sum"};
        for (int k = 0; k < lambdas.length; k++) {
            StructuralHillClimb hc = new StructuralHillClimb(w);
            hc.skelWeight = lambdas[k];
            long t = System.currentTimeMillis();
            Dag g = hc.search(OrientationChooser.majoritySkeletonOriented(w), TreewidthBudget.unbounded());
            row.emit("HC-maj-" + names[k], "-", g, wTime + (System.currentTimeMillis() - t) / 1000.0);
        }
    }

    /**
     * Hard treewidth budgets, for the question "what does tractability cost?".
     * Only methods that respect any budget from their own start take part: the
     * climb from the empty graph, MMG, and Greedy-Ea. The G+ and majority starts
     * would first need a repair step whenever they exceed the budget.
     */
    static void runBudget(List<Dag> dags, PairWeights w, Row row, double wTime) {
        int twMax = Utils.getTreeWidth(new Dag(ConsensusUnion.fusionUnion(dags)));
        double lex = 1.0 / (2 * w.r());
        List<Integer> grid = new ArrayList<>();
        for (int b : new int[]{2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20}) if (b < twMax) grid.add(b);
        grid.add(twMax);
        for (int b : grid) {
            TreewidthBudget tb = new TreewidthBudget(b);
            StructuralHillClimb s = new StructuralHillClimb(w);
            s.skelWeight = lex;
            long t = System.currentTimeMillis();
            Dag g = s.search(new Dag(w.nodes()), tb);
            row.emit("HC-empty", "b" + b + "/" + twMax, g, wTime + (System.currentTimeMillis() - t) / 1000.0);

            MajorityGreedy m = new MajorityGreedy(dags);
            m.fastCandidateUpdate = true;
            t = System.currentTimeMillis();
            g = m.search(tb);
            row.emit("MMG", "b" + b + "/" + twMax, g, wTime + (System.currentTimeMillis() - t) / 1000.0);

            t = System.currentTimeMillis();
            g = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(b));
            row.emit("Greedy-Ea", "b" + b + "/" + twMax, g, (System.currentTimeMillis() - t) / 1000.0);
        }
    }

    /**
     * MCBNC under hard treewidth budgets, on the grid of runBudget. MCBNC deletes
     * from the union until the treewidth fits, so one run down to the tightest
     * budget passes through all of them: with experiments_tw it records the graph
     * each time the max clique size drops by one, and snapshot i is the first
     * graph with max clique size at most tw(G+) - 1 - i. Each row is charged the
     * time until its snapshot, which includes the construction of the union.
     */
    static void runMcbncBudget(List<Dag> dags, PairWeights w, Row row) {
        long t = System.currentTimeMillis();
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2);
        double tGp = (System.currentTimeMillis() - t) / 1000.0;
        int twMax = Utils.getTreeWidth(mc.fusionUnion);
        Dag union = new Dag(mc.fusionUnion);
        mc.experiments_tw = true;
        mc.fusion();
        List<Integer> grid = new ArrayList<>();
        for (int b : new int[]{2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20}) if (b < twMax) grid.add(b);
        for (int b : grid) {
            int idx = twMax - 1 - b;
            if (idx >= mc.outputExperimentDAGs.size()) continue;      // the sweep stopped above this budget
            row.emit("MCBNC", "b" + b + "/" + twMax, mc.outputExperimentDAGs.get(idx),
                    tGp + mc.outputExperimentTimes.get(idx));
        }
        row.emit("MCBNC", "b" + twMax + "/" + twMax, union, tGp);
    }

    /**
     * Objectives built from the distance between equivalence classes (skeleton
     * plus immoralities), alone or combined with the moral one. Weights are
     * (moral, skeleton, immorality).
     */
    static void runClassVariants(List<Dag> dags, PairWeights w, Row row, double wTime) {
        long t0 = System.currentTimeMillis();
        ImmoralityWeights imm = new ImmoralityWeights(dags, w);
        double iTime = (System.currentTimeMillis() - t0) / 1000.0;
        Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
        double eps = 1.0 / (2 * w.r());
        double tiny = 1e-3 / w.r();
        String[] names = {"lex", "lexC", "class", "classM", "sum"};
        double[][] cfg = {{1, eps, 0}, {1, tiny, tiny}, {0, 1, 1}, {tiny, 1, 1}, {1, 1, 1}};
        for (int k = 0; k < names.length; k++) {
            for (String start : new String[]{"empty", "maj", "gplus"}) {
                StructuralHillClimb s = new StructuralHillClimb(w);
                s.moralWeight = cfg[k][0];
                s.skelWeight = cfg[k][1];
                s.imm = imm;
                s.immWeight = cfg[k][2];
                Dag init = switch (start) {
                    case "empty" -> new Dag(w.nodes());
                    case "maj" -> OrientationChooser.majoritySkeletonOriented(w);
                    default -> new Dag(gp);
                };
                long t = System.currentTimeMillis();
                Dag g = s.search(init, TreewidthBudget.unbounded());
                row.emit("C-" + start + "-" + names[k], "-", g, wTime + iTime + (System.currentTimeMillis() - t) / 1000.0);
            }
        }
    }

    /**
     * Treewidth one, where the optimum is known exactly (a minimum-weight
     * spanning forest on the negative moral weights, oriented as a branching),
     * against the heuristics run under the same bound (max clique size 2).
     */
    static void runTw1(List<Dag> dags, PairWeights w, Row row, double wTime) {
        long t = System.currentTimeMillis();
        Dag opt = kruskalForest(w);
        row.emit("Kruskal", "b2", opt, wTime + (System.currentTimeMillis() - t) / 1000.0);

        TreewidthBudget tb = new TreewidthBudget(2);
        StructuralHillClimb s = new StructuralHillClimb(w);
        s.skelWeight = 1.0 / (2 * w.r());
        t = System.currentTimeMillis();
        Dag g = s.search(new Dag(w.nodes()), tb);
        row.emit("HC-empty", "b2", g, wTime + (System.currentTimeMillis() - t) / 1000.0);

        MajorityGreedy m = new MajorityGreedy(dags);
        m.fastCandidateUpdate = true;
        t = System.currentTimeMillis();
        g = m.search(tb);
        row.emit("MMG", "b2", g, wTime + (System.currentTimeMillis() - t) / 1000.0);

        t = System.currentTimeMillis();
        g = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", "2");
        row.emit("Greedy-Ea", "b2", g, (System.currentTimeMillis() - t) / 1000.0);
    }

    /** Kruskal on the pairs of negative moral weight, each tree oriented away from a root. */
    public static Dag kruskalForest(PairWeights w) {
        int n = w.n();
        List<int[]> cand = new ArrayList<>();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (w.wMoral(i, j) < 0) cand.add(new int[]{i, j});
        cand.sort(Comparator.comparingDouble(p -> w.wMoral(p[0], p[1])));
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] p : cand) {
            int a = find(parent, p[0]), b = find(parent, p[1]);
            if (a == b) continue;
            parent[a] = b;
            adj.get(p[0]).add(p[1]);
            adj.get(p[1]).add(p[0]);
        }
        Dag out = new Dag(w.nodes());
        boolean[] seen = new boolean[n];
        for (int root = 0; root < n; root++) {
            if (seen[root]) continue;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(root);
            seen[root] = true;
            while (!queue.isEmpty()) {
                int v = queue.poll();
                for (int u : adj.get(v)) {
                    if (seen[u]) continue;
                    seen[u] = true;
                    out.addEdge(Edges.directedEdge(w.nodes().get(v), w.nodes().get(u)));
                    queue.add(u);
                }
            }
        }
        return out;
    }

    static int find(int[] parent, int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }

    /**
     * Every selection redone with the criterion that matches the SHD evaluation:
     * the mean SHD between CPDAGs to the inputs, instead of the mean moral distance.
     * This answers whether the SHD results depend on selecting by SMHD. PHC-G+ has
     * no selection, so its SHD-oriented counterpart changes the objective instead:
     * skeleton plus immoralities (the decomposable part of the class distance),
     * with the moral term as a tie-breaker. Method names carry an "@shd" suffix.
     */
    static void runShdSelection(List<Dag> dags, PairWeights w, Row row, double wTime) {
        long t0 = System.currentTimeMillis();
        List<Graph> in = new ArrayList<>();
        for (Dag d : dags) in.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        CpdagShdCounts counts = new CpdagShdCounts(in);
        double inTime = (System.currentTimeMillis() - t0) / 1000.0;
        java.util.function.ToDoubleFunction<Dag> shdIn = g -> counts.mean(GraphTransforms.dagToCpdag(new EdgeListGraph(g)));

        // Medoid by SHD
        long t = System.currentTimeMillis();
        Dag med = null; double best = Double.POSITIVE_INFINITY;
        for (Dag d : dags) { double v = shdIn.applyAsDouble(d); if (v < best) { best = v; med = d; } }
        row.emit("medoid@shd", "-", med, inTime + (System.currentTimeMillis() - t) / 1000.0);

        // PHC starts, selected by SHD
        double lex = 1.0 / (2 * w.r());
        Map<String, Dag> hc = new LinkedHashMap<>();
        Map<String, Double> hcTime = new LinkedHashMap<>();
        hcTime.put("HC-empty", climb(w, new Dag(w.nodes()), lex, row, "HC-empty@shd-run", wTime, hc));
        hcTime.put("HC-maj", climb(w, OrientationChooser.majoritySkeletonOriented(w), lex, row, "HC-maj@shd-run", wTime, hc));
        emitBestBy(shdIn, row, "HC-best-free@shd", hc, hcTime, wTime, inTime);
        t = System.currentTimeMillis();
        Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
        double tGp = (System.currentTimeMillis() - t) / 1000.0;
        hcTime.put("HC-gplus", climb(w, new Dag(gp), lex, row, "HC-gplus@shd-run", wTime + tGp, hc));
        emitBestBy(shdIn, row, "HC-best@shd", hc, hcTime, wTime, inTime);

        // PHC-G+ with the SHD-oriented objective: skeleton + immoralities, moral as tie-break
        t = System.currentTimeMillis();
        ImmoralityWeights imm = new ImmoralityWeights(dags, w);
        StructuralHillClimb s = new StructuralHillClimb(w);
        s.moralWeight = 1e-3 / w.r();
        s.skelWeight = 1.0;
        s.imm = imm;
        s.immWeight = 1.0;
        Dag cls = s.search(new Dag(gp), TreewidthBudget.unbounded());
        row.emit("HC-gplus@shd", "-", cls, wTime + tGp + (System.currentTimeMillis() - t) / 1000.0);

        // Greedy-Ea: treewidth sweep selected by SHD
        t = System.currentTimeMillis();
        int twMax = Utils.getTreeWidth(gp);
        Dag bestG = null; best = Double.POSITIVE_INFINITY; int bestB = -1;
        for (int b : budgetGrid(twMax)) {
            Dag g = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(b));
            double v = shdIn.applyAsDouble(g);
            if (v < best) { best = v; bestG = g; bestB = b; }
        }
        row.emit("Greedy-Ea@shd", "tw" + bestB + "/" + twMax, bestG, inTime + tGp + (System.currentTimeMillis() - t) / 1000.0);

        // MCBNC: theta trajectory selected by SHD
        t = System.currentTimeMillis();
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY);
        mc.experiments_perc = true;
        mc.fusion();
        Dag bestM = null; best = Double.POSITIVE_INFINITY; int bestIdx = -1, idx = 0;
        for (Dag snap : mc.outputExperimentDAGs) {
            double v = shdIn.applyAsDouble(snap);
            if (v < best) { best = v; bestM = snap; bestIdx = idx; }
            idx++;
        }
        if (bestM != null)
            row.emit("MCBNC@shd", "snap" + bestIdx + "/" + idx, bestM, inTime + (System.currentTimeMillis() - t) / 1000.0);
    }

    /**
     * PHC with a trajectory, like MCBNC: every point of the descent (and the
     * start) is scored by the mean CPDAG-SHD to the inputs and the best one is
     * returned. The objective still drives the search; only the returned point
     * changes, which is exactly what theta selection does for MCBNC.
     */
    static void runTrajectory(List<Dag> dags, PairWeights w, Row row, double wTime) {
        List<Graph> in = new ArrayList<>();
        for (Dag d : dags) in.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        CpdagShdCounts counts = new CpdagShdCounts(in);
        java.util.function.ToDoubleFunction<Dag> shdIn = g -> counts.mean(GraphTransforms.dagToCpdag(new EdgeListGraph(g)));
        double lex = 1.0 / (2 * w.r());
        long t = System.currentTimeMillis();
        Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
        double tGp = (System.currentTimeMillis() - t) / 1000.0;

        Map<String, Dag> starts = new LinkedHashMap<>();
        // The descent from G+ is not scored: in the pilot its trajectory never beat
        // its endpoint under SHD, and scoring every step of it is the expensive part.
        starts.put("empty", new Dag(w.nodes()));
        starts.put("maj", OrientationChooser.majoritySkeletonOriented(w));
        Map<String, Dag> bestOf = new LinkedHashMap<>();
        Map<String, Double> bestVal = new LinkedHashMap<>();
        Map<String, Double> took = new LinkedHashMap<>();
        for (Map.Entry<String, Dag> e : starts.entrySet()) {
            final Dag[] best = {new Dag(e.getValue())};
            final double[] bv = {shdIn.applyAsDouble(e.getValue())};
            StructuralHillClimb s = new StructuralHillClimb(w);
            s.skelWeight = lex;
            s.onStep = g -> {
                double v = shdIn.applyAsDouble(g);
                if (v < bv[0]) { bv[0] = v; best[0] = new Dag(g); }
            };
            t = System.currentTimeMillis();
            s.search(new Dag(e.getValue()), TreewidthBudget.unbounded());
            double time = wTime + (e.getKey().equals("gplus") ? tGp : 0) + (System.currentTimeMillis() - t) / 1000.0;
            bestOf.put(e.getKey(), best[0]);
            bestVal.put(e.getKey(), bv[0]);
            took.put(e.getKey(), time);
        }
        String fk = bestVal.get("empty") <= bestVal.get("maj") ? "empty" : "maj";
        row.emit("HC-best-free@traj", fk, bestOf.get(fk), took.get("empty") + took.get("maj") - wTime);
    }

    /** Like emitBest, but the data-free criterion is a caller-supplied distance to the inputs. */
    static void emitBestBy(java.util.function.ToDoubleFunction<Dag> crit, Row row, String name,
                           Map<String, Dag> hc, Map<String, Double> times, double wTime, double extra) {
        String bestK = null; double bestV = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, Dag> e : hc.entrySet()) {
            double v = crit.applyAsDouble(e.getValue());
            if (v < bestV) { bestV = v; bestK = e.getKey(); }
        }
        double total = extra - wTime * (times.size() - 1);
        for (double x : times.values()) total += x;
        row.emit(name, bestK, hc.get(bestK), total);
    }

    static void runMcbnc(List<Dag> dags, PairWeights w, Row row) {
        long t = System.currentTimeMillis();
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY);
        mc.experiments_perc = true;
        mc.fusion();
        Dag best = null; double bestVal = Double.POSITIVE_INFINITY; int bestIdx = -1, idx = 0;
        for (Dag snap : mc.outputExperimentDAGs) {
            double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(snap)));
            if (v < bestVal) { bestVal = v; best = snap; bestIdx = idx; }
            idx++;
        }
        double time = (System.currentTimeMillis() - t) / 1000.0;
        if (best != null) row.emit("MCBNC", "snap" + bestIdx + "/" + idx, best, time);
    }

    /**
     * The first point of the MCBNC theta trajectory (the union after its zero-cost
     * deletions), which the sweep before the fix never recorded. The row is
     * "MCBNC-first"; param is "<missing>:<mean CPDAG-SHD to the inputs>", where
     * missing=0 means the sweep ended before any positive cost, so the old
     * trajectory already held this graph. Lets the old rows be corrected exactly
     * under the moral protocol without repeating the sweep.
     */
    static void runMcbncFirst(List<Dag> dags, Row row) {
        long t = System.currentTimeMillis();
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY);
        mc.experiments_perc = true;
        mc.stopAtFirstPoint = true;
        mc.fusion();
        Dag f = mc.outputExperimentDAGs.get(0);
        double time = (System.currentTimeMillis() - t) / 1000.0;
        Graph c = GraphTransforms.dagToCpdag(new EdgeListGraph(f));
        double v = 0;
        for (Dag d : dags) v += Utils.SHDundir(c, GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        row.emit("MCBNC-first", (mc.stoppedAtFirstPoint ? 1 : 0) + ":" + String.format(java.util.Locale.ROOT, "%.4f", v / dags.size()), f, time);
    }

    /**
     * One theta sweep of MCBNC, selected under both protocols: by D_M (MCBNC) and
     * by the mean CPDAG-SHD to the inputs (MCBNC@shd). Each row is charged the
     * sweep plus its own scoring, and MCBNC@shd also the conversion of the inputs.
     */
    static void runMcbncBoth(List<Dag> dags, PairWeights w, Row row) {
        long t0 = System.currentTimeMillis();
        List<Graph> in = new ArrayList<>();
        for (Dag d : dags) in.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        CpdagShdCounts counts = new CpdagShdCounts(in);
        double inTime = (System.currentTimeMillis() - t0) / 1000.0;
        long t = System.currentTimeMillis();
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY);
        mc.experiments_perc = true;
        mc.fusion();
        double sweep = (System.currentTimeMillis() - t) / 1000.0;
        int n = mc.outputExperimentDAGs.size();

        t = System.currentTimeMillis();
        Dag best = null; double bestVal = Double.POSITIVE_INFINITY; int bestIdx = -1;
        for (int i = 0; i < n; i++) {
            Dag snap = mc.outputExperimentDAGs.get(i);
            double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(snap)));
            if (v < bestVal) { bestVal = v; best = snap; bestIdx = i; }
        }
        if (best != null) row.emit("MCBNC", "snap" + bestIdx + "/" + n, best, sweep + (System.currentTimeMillis() - t) / 1000.0);

        t = System.currentTimeMillis();
        best = null; bestVal = Double.POSITIVE_INFINITY; bestIdx = -1;
        for (int i = 0; i < n; i++) {
            Dag snap = mc.outputExperimentDAGs.get(i);
            double v = counts.total(GraphTransforms.dagToCpdag(new EdgeListGraph(snap)));
            if (v < bestVal) { bestVal = v; best = snap; bestIdx = i; }
        }
        if (best != null) row.emit("MCBNC@shd", "snap" + bestIdx + "/" + n, best, inTime + sweep + (System.currentTimeMillis() - t) / 1000.0);
    }

    // -----------------------------------------------------------------------
    // Shared union. On the large networks G+ dominates the cost of every method
    // that starts from it, so it is built once per replicate (group gplus), stored
    // as an arc list next to its build time, and read back by the groups that
    // start from it (phcg, mcg), the way the GES DAGs of the clients are cached.
    // Their rows are still charged the stored build time, so the time column
    // keeps its meaning and the search time is the difference with the G+ row.
    // -----------------------------------------------------------------------

    static File unionFile(String net, int r, int rep, String tag) {
        String dir = System.getProperty("gplusDir", "./results/greedy/gplus");
        return new File(dir, "gplus-n" + tag + "-" + net + "-r" + r + "-rep" + rep + ".txt");
    }

    /** Builds the union the way MCBNC does and stores it with its build time. */
    static void buildUnion(List<Dag> dags, PairWeights w, File f) throws Exception {
        long t = System.currentTimeMillis();
        Dag gp = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY).fusionUnion;
        double seconds = (System.currentTimeMillis() - t) / 1000.0;
        f.getParentFile().mkdirs();
        File tmp = new File(f.getPath() + ".tmp");
        try (PrintWriter pw = new PrintWriter(new FileWriter(tmp))) {
            pw.println(String.format(Locale.US, "%.4f", seconds));
            for (Edge e : gp.getEdges()) {
                Node x = e.getNode1(), y = e.getNode2();
                if (!gp.isParentOf(x, y)) { Node s = x; x = y; y = s; }
                pw.println(x.getName() + "\t" + y.getName());
            }
        }
        if (!tmp.renameTo(f)) throw new IllegalStateException("cannot write " + f);
    }

    /** Reads a stored union onto the nodes of the inputs; null when it has not been built. */
    static Object[] readUnion(PairWeights w, File f) throws Exception {
        if (!f.exists()) return null;
        Map<String, Node> byName = new HashMap<>();
        for (Node v : w.nodes()) byName.put(v.getName(), v);
        Dag gp = new Dag(w.nodes());
        double seconds;
        try (Scanner sc = new Scanner(f)) {
            seconds = Double.parseDouble(sc.nextLine().trim());
            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                if (line.isEmpty()) continue;
                String[] p = line.split("\t");
                gp.addEdge(Edges.directedEdge(byName.get(p[0]), byName.get(p[1])));
            }
        }
        return new Object[]{gp, seconds};
    }

    static void runUnion(List<Dag> dags, PairWeights w, Row row, String net, int r, int rep, String tag) throws Exception {
        File f = unionFile(net, r, rep, tag);
        Object[] u = readUnion(w, f);
        if (u == null) {
            buildUnion(dags, w, f);
            u = readUnion(w, f);          // always on the nodes of the inputs, like the later groups
        }
        row.emit("G+", "-", (Dag) u[0], (Double) u[1]);
    }

    static Object[] storedUnion(PairWeights w, String net, int r, int rep, String tag) throws Exception {
        File f = unionFile(net, r, rep, tag);
        Object[] u = readUnion(w, f);
        if (u == null) throw new IllegalStateException("union not built yet: " + f);
        return u;
    }

    /** PHC-G+ from the stored union. */
    static void runPhcFromUnion(List<Dag> dags, PairWeights w, Row row, double wTime,
                                String net, int r, int rep, String tag) throws Exception {
        Object[] u = storedUnion(w, net, r, rep, tag);
        climb(w, new Dag((Dag) u[0]), 1.0 / (2 * w.r()), row, "HC-gplus", wTime + (Double) u[1], null);
    }

    /** MCBNC from the stored union, selected under both protocols as in runMcbncBoth. */
    static void runMcbncFromUnion(List<Dag> dags, PairWeights w, Row row,
                                  String net, int r, int rep, String tag) throws Exception {
        Object[] u = storedUnion(w, net, r, rep, tag);
        double tGp = (Double) u[1];
        long t0 = System.currentTimeMillis();
        List<Graph> in = new ArrayList<>();
        for (Dag d : dags) in.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        CpdagShdCounts counts = new CpdagShdCounts(in);
        double inTime = (System.currentTimeMillis() - t0) / 1000.0;
        long t = System.currentTimeMillis();
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, new Dag((Dag) u[0]), 10, 2, Double.POSITIVE_INFINITY);
        mc.experiments_perc = true;
        mc.fusion();
        double sweep = tGp + (System.currentTimeMillis() - t) / 1000.0;
        int n = mc.outputExperimentDAGs.size();

        t = System.currentTimeMillis();
        Dag best = null; double bestVal = Double.POSITIVE_INFINITY; int bestIdx = -1;
        for (int i = 0; i < n; i++) {
            Dag snap = mc.outputExperimentDAGs.get(i);
            double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(snap)));
            if (v < bestVal) { bestVal = v; best = snap; bestIdx = i; }
        }
        if (best != null) row.emit("MCBNC", "snap" + bestIdx + "/" + n, best, sweep + (System.currentTimeMillis() - t) / 1000.0);

        t = System.currentTimeMillis();
        best = null; bestVal = Double.POSITIVE_INFINITY; bestIdx = -1;
        for (int i = 0; i < n; i++) {
            Dag snap = mc.outputExperimentDAGs.get(i);
            double v = counts.total(GraphTransforms.dagToCpdag(new EdgeListGraph(snap)));
            if (v < bestVal) { bestVal = v; best = snap; bestIdx = i; }
        }
        if (best != null) row.emit("MCBNC@shd", "snap" + bestIdx + "/" + n, best, inTime + sweep + (System.currentTimeMillis() - t) / 1000.0);
    }

    /**
     * Greedy-Ea run the way MCBNC is run: sweep its control parameter (the
     * treewidth budget) and keep the output closest to the inputs. With no budget
     * it adds every arc and returns G+, so a single unbudgeted run is not the
     * method; neither is one arbitrary budget.
     */
    static void runGreedyEa(List<Dag> dags, PairWeights w, Row row) {
        long t = System.currentTimeMillis();
        Dag gp = new Dag(ConsensusUnion.fusionUnion(dags));
        int twMax = Utils.getTreeWidth(gp);
        Dag best = null; double bestVal = Double.POSITIVE_INFINITY; int bestB = -1;
        for (int b : budgetGrid(twMax)) {
            Dag g = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(b));
            double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
            if (v < bestVal) { bestVal = v; best = g; bestB = b; }
        }
        double time = (System.currentTimeMillis() - t) / 1000.0;
        row.emit("Greedy-Ea", "tw" + bestB + "/" + twMax, best, time);
    }

    // -----------------------------------------------------------------------
    // Revision groups (2026-10-14): a third start for PHC-Sel (the medoid), the
    // wider candidate pool, the weight lambda of the skeleton term, the averaged
    // network of bnlearn, PHC-Sel under a treewidth bound, and GES on the pooled
    // data as a reference that a data-free method cannot use.
    // -----------------------------------------------------------------------

    /** The input closest to the others under D_M. */
    static Dag medoidMoral(PairWeights w, List<Dag> dags) {
        Dag best = null; double bv = Double.POSITIVE_INFINITY;
        for (Dag d : dags) {
            double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(d)));
            if (v < bv) { bv = v; best = d; }
        }
        return best;
    }

    /** A copy of g on the node objects of the weights, so the trackers can use it. */
    static Dag onNodes(PairWeights w, Graph g) {
        Map<String, Node> byName = new HashMap<>();
        for (Node v : w.nodes()) byName.put(v.getName(), v);
        Dag out = new Dag(w.nodes());
        for (Edge e : g.getEdges()) {
            Node a = Edges.getDirectedEdgeTail(e), b = Edges.getDirectedEdgeHead(e);
            out.addEdge(Edges.directedEdge(byName.get(a.getName()), byName.get(b.getName())));
        }
        return out;
    }

    static double climbCfg(PairWeights w, Dag start, double lambda, boolean pool, TreewidthBudget tb,
                           Row row, String name, String param, double overhead, Map<String, Dag> keep) {
        StructuralHillClimb s = new StructuralHillClimb(w);
        s.skelWeight = lambda;
        s.poolMoral = pool;
        long t = System.currentTimeMillis();
        Dag g = s.search(start, tb);
        double time = overhead + (System.currentTimeMillis() - t) / 1000.0;
        if (row != null) row.emit(name, param, g, time);
        if (keep != null) keep.put(name, g);
        return time;
    }

    /**
     * Moral protocol. HC-medoid starts from the medoid; HC-best-free3 selects by
     * D_M among the climbs from the empty graph, the majority skeleton and the
     * medoid. The same selection is repeated with the wider pool ("-pool", also
     * over the two original starts as HC-best-free-pool) and with the skeleton
     * weight lambda in {0, 1/2, 1} instead of 1/(2r) ("-l0", "-lhalf", "-l1").
     * The averaged network has no selection; its param is the threshold.
     */
    static void runRevision(List<Dag> dags, PairWeights w, Dag gold, Graph goldCpdag, Row row, double wTime) {
        long t = System.currentTimeMillis();
        AveragedNetwork an = new AveragedNetwork();
        Dag avg = an.build(w);
        row.emit("avgNet", String.format(Locale.US, "%.2f", an.threshold), avg, wTime + (System.currentTimeMillis() - t) / 1000.0);

        t = System.currentTimeMillis();
        Dag med = medoidMoral(w, dags);
        double tMed = (System.currentTimeMillis() - t) / 1000.0;
        double lex = 1.0 / (2 * w.r());
        Object[][] cfg = {{"", lex, false}, {"-pool", lex, true}, {"-l0", 0.0, false}, {"-lhalf", 0.5, false}, {"-l1", 1.0, false}};
        for (Object[] c : cfg) {
            String sfx = (String) c[0];
            double lambda = (Double) c[1];
            boolean pool = (Boolean) c[2];
            boolean emitStarts = sfx.isEmpty() || sfx.equals("-pool");
            Map<String, Dag> hc = new LinkedHashMap<>();
            Map<String, Double> times = new LinkedHashMap<>();
            Row r = emitStarts ? row : null;
            times.put("HC-empty" + sfx, climbCfg(w, new Dag(w.nodes()), lambda, pool, TreewidthBudget.unbounded(), r, "HC-empty" + sfx, "-", wTime, hc));
            times.put("HC-maj" + sfx, climbCfg(w, OrientationChooser.majoritySkeletonOriented(w), lambda, pool, TreewidthBudget.unbounded(), r, "HC-maj" + sfx, "-", wTime, hc));
            if (pool) emitBest(w, row, "HC-best-free-pool", hc, times, wTime);
            times.put("HC-medoid" + sfx, climbCfg(w, onNodes(w, med), lambda, pool, TreewidthBudget.unbounded(), r, "HC-medoid" + sfx, "-", wTime + tMed, hc));
            emitBest(w, row, "HC-best-free3" + sfx, hc, times, wTime);
        }
    }

    /**
     * Structural protocol, as runTrajectory: each climb is scored at every step by
     * the mean CPDAG-SHD to the inputs and the best point is kept, now from three
     * starts (empty, majority skeleton, medoid by that distance), with the original
     * pool (HC-best-free3@traj) and the wider one (HC-best-free3-pool@traj).
     */
    static void runRevisionTrajectory(List<Dag> dags, PairWeights w, Row row, double wTime) {
        long t0 = System.currentTimeMillis();
        List<Graph> in = new ArrayList<>();
        for (Dag d : dags) in.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        CpdagShdCounts counts = new CpdagShdCounts(in);
        java.util.function.ToDoubleFunction<Dag> shdIn = g -> counts.mean(GraphTransforms.dagToCpdag(new EdgeListGraph(g)));
        Dag med = null; double bm = Double.POSITIVE_INFINITY;
        for (Dag d : dags) { double v = shdIn.applyAsDouble(d); if (v < bm) { bm = v; med = d; } }
        double tPre = (System.currentTimeMillis() - t0) / 1000.0;
        double lex = 1.0 / (2 * w.r());

        for (boolean pool : new boolean[]{false, true}) {
            Map<String, Dag> starts = new LinkedHashMap<>();
            starts.put("empty", new Dag(w.nodes()));
            starts.put("maj", OrientationChooser.majoritySkeletonOriented(w));
            starts.put("medoid", onNodes(w, med));
            String bestK = null; Dag bestG = null; double bestV = Double.POSITIVE_INFINITY, total = wTime + tPre;
            for (Map.Entry<String, Dag> e : starts.entrySet()) {
                final Dag[] best = {new Dag(e.getValue())};
                final double[] bv = {shdIn.applyAsDouble(e.getValue())};
                StructuralHillClimb s = new StructuralHillClimb(w);
                s.skelWeight = lex;
                s.poolMoral = pool;
                s.onStep = g -> {
                    double v = shdIn.applyAsDouble(g);
                    if (v < bv[0]) { bv[0] = v; best[0] = new Dag(g); }
                };
                long t = System.currentTimeMillis();
                s.search(new Dag(e.getValue()), TreewidthBudget.unbounded());
                total += (System.currentTimeMillis() - t) / 1000.0;
                if (bv[0] < bestV) { bestV = bv[0]; bestK = e.getKey(); bestG = best[0]; }
            }
            row.emit(pool ? "HC-best-free3-pool@traj" : "HC-best-free3@traj", bestK, bestG, total);
        }
    }

    /**
     * The averaged network as a fourth start. Moral protocol: HC-avg (climb from it) and
     * HC-best-free4 (selection by D_M among empty, majority, medoid and averaged starts).
     * Structural protocol: HC-avg@traj and HC-best-free4@traj, trajectory selection by the mean
     * CPDAG-SHD to the inputs, the start itself included as a candidate.
     */
    static void runAvgStart(List<Dag> dags, PairWeights w, Row row, double wTime) {
        long t0 = System.currentTimeMillis();
        List<Graph> in = new ArrayList<>();
        for (Dag d : dags) in.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        CpdagShdCounts counts = new CpdagShdCounts(in);
        java.util.function.ToDoubleFunction<Dag> shdIn = g -> counts.mean(GraphTransforms.dagToCpdag(new EdgeListGraph(g)));
        Dag medS = null; double bm = Double.POSITIVE_INFINITY;
        for (Dag d : dags) { double v = shdIn.applyAsDouble(d); if (v < bm) { bm = v; medS = d; } }
        double tPre = (System.currentTimeMillis() - t0) / 1000.0;
        double lex = 1.0 / (2 * w.r());
        Dag avg = new AveragedNetwork().build(w);

        Map<String, Dag> hc = new LinkedHashMap<>();
        Map<String, Double> times = new LinkedHashMap<>();
        times.put("HC-empty", climbCfg(w, new Dag(w.nodes()), lex, false, TreewidthBudget.unbounded(), null, "HC-empty", "-", wTime, hc));
        times.put("HC-maj", climbCfg(w, OrientationChooser.majoritySkeletonOriented(w), lex, false, TreewidthBudget.unbounded(), null, "HC-maj", "-", wTime, hc));
        times.put("HC-medoid", climbCfg(w, onNodes(w, medoidMoral(w, dags)), lex, false, TreewidthBudget.unbounded(), null, "HC-medoid", "-", wTime, hc));
        times.put("HC-avg", climbCfg(w, new Dag(avg), lex, false, TreewidthBudget.unbounded(), row, "HC-avg", "-", wTime, hc));
        emitBest(w, row, "HC-best-free4", hc, times, wTime);

        Map<String, Dag> starts = new LinkedHashMap<>();
        starts.put("empty", new Dag(w.nodes()));
        starts.put("maj", OrientationChooser.majoritySkeletonOriented(w));
        starts.put("medoid", onNodes(w, medS));
        starts.put("avg", new Dag(avg));
        String bestK = null; Dag bestG = null; double bestV = Double.POSITIVE_INFINITY, total = wTime + tPre;
        for (Map.Entry<String, Dag> e : starts.entrySet()) {
            final Dag[] best = {new Dag(e.getValue())};
            final double[] bv = {shdIn.applyAsDouble(e.getValue())};
            StructuralHillClimb s = new StructuralHillClimb(w);
            s.skelWeight = lex;
            s.onStep = g -> {
                double v = shdIn.applyAsDouble(g);
                if (v < bv[0]) { bv[0] = v; best[0] = new Dag(g); }
            };
            long t = System.currentTimeMillis();
            s.search(new Dag(e.getValue()), TreewidthBudget.unbounded());
            total += (System.currentTimeMillis() - t) / 1000.0;
            if (e.getKey().equals("avg")) row.emit("HC-avg@traj", "-", best[0], wTime + tPre);
            if (bv[0] < bestV) { bestV = bv[0]; bestK = e.getKey(); bestG = best[0]; }
        }
        row.emit("HC-best-free4@traj", bestK, bestG, total);
    }

    /**
     * Removes arcs, least harmful first under the objective of the climb (moral
     * plus lambda times skeleton), until the graph fits the budget. Gives the
     * majority and medoid starts a feasible point under any bound.
     */
    static Dag pruneToBudget(PairWeights w, Dag g, TreewidthBudget tb, double lambda) {
        MoralGraphTracker t = new MoralGraphTracker(w, g);
        while (!tb.fits(g)) {
            Node bx = null, by = null; double bestGain = Double.POSITIVE_INFINITY;
            for (Edge e : new ArrayList<>(g.getEdges())) {
                Node x = Edges.getDirectedEdgeTail(e), y = Edges.getDirectedEdgeHead(e);
                double gain = t.gainRemoveArc(x, y) - lambda * w.wSkel(w.idx(x), w.idx(y));
                if (gain < bestGain) { bestGain = gain; bx = x; by = y; }
            }
            if (bx == null) break;
            t.removeArc(bx, by);
        }
        return g;
    }

    /**
     * PHC-Sel under the treewidth budgets of runBudget: the majority and medoid
     * starts are first pruned to the budget, then every start is climbed under it
     * and the output closest to the inputs (D_M) is returned as HC-best-free3.
     * The pruned starts are also reported alone (HC-maj-pruned, HC-medoid-pruned). With pool the
     * insertions range over P^M and every name carries "-pool" (group budget2p).
     */
    static void runBudgetSel(List<Dag> dags, PairWeights w, Row row, double wTime, boolean pool) {
        String sfx = pool ? "-pool" : "";
        int twMax = Utils.getTreeWidth(new Dag(ConsensusUnion.fusionUnion(dags)));
        double lex = 1.0 / (2 * w.r());
        List<Integer> grid = new ArrayList<>();
        for (int b : new int[]{2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20}) if (b < twMax) grid.add(b);
        grid.add(twMax);
        Dag med = onNodes(w, medoidMoral(w, dags));
        Dag maj = OrientationChooser.majoritySkeletonOriented(w);
        for (int b : grid) {
            TreewidthBudget tb = new TreewidthBudget(b);
            String param = "b" + b + "/" + twMax;
            Map<String, Dag> hc = new LinkedHashMap<>();
            Map<String, Double> times = new LinkedHashMap<>();
            times.put("HC-empty", climbCfg(w, new Dag(w.nodes()), lex, pool, tb, null, "HC-empty", param, wTime, hc));
            long t = System.currentTimeMillis();
            Dag pm = pruneToBudget(w, new Dag(maj), tb, lex);
            double tp = (System.currentTimeMillis() - t) / 1000.0;
            times.put("HC-maj-pruned", climbCfg(w, pm, lex, pool, tb, row, "HC-maj-pruned" + sfx, param, wTime + tp, hc));
            t = System.currentTimeMillis();
            Dag pd = pruneToBudget(w, new Dag(med), tb, lex);
            tp = (System.currentTimeMillis() - t) / 1000.0;
            times.put("HC-medoid-pruned", climbCfg(w, pd, lex, pool, tb, row, "HC-medoid-pruned" + sfx, param, wTime + tp, hc));
            String bestK = null; double bestV = Double.POSITIVE_INFINITY;
            for (Map.Entry<String, Dag> e : hc.entrySet()) {
                double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(e.getValue())))
                        + w.skelOf(w.objectiveOfSkeleton(e.getValue())) * 1e-6;
                if (v < bestV) { bestV = v; bestK = e.getKey(); }
            }
            double total = -wTime * (times.size() - 1);
            for (double x : times.values()) total += x;
            row.emit("HC-best-free3" + sfx, param, hc.get(bestK), total);
        }
    }

    /**
     * GES on the pooled data of the r clients of the replicate (the first
     * sampleSize rows of each), the reference a data-free method gives up. The
     * rows are concatenated as text and read once, so every variable keeps one
     * category list.
     */
    static void runCentral(PairWeights w, Row row, String net, int r, int rep, int sampleSize) throws Exception {
        File tmp = File.createTempFile("central-" + net + "-", ".csv");
        tmp.deleteOnExit();
        try (PrintWriter pw = new PrintWriter(new FileWriter(tmp))) {
            for (int i = rep * r; i < (rep + 1) * r; i++) {
                try (Scanner sc = new Scanner(new File("./res/networks/BBDD/" + net + "/" + net + "." + i + ".csv"))) {
                    String header = sc.nextLine();
                    if (i == rep * r) pw.println(header);
                    for (int k = 0; k < sampleSize && sc.hasNextLine(); k++) pw.println(sc.nextLine());
                }
            }
        }
        long t = System.currentTimeMillis();
        edu.cmu.tetrad.data.DataSet data = Utils.readData(tmp.getPath());
        Graph g = new org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder(data, true).search();
        double time = (System.currentTimeMillis() - t) / 1000.0;
        tmp.delete();
        row.emit("GES-central", String.valueOf(data.getNumRows()), onNodes(w, new Dag(g)), time);
    }

    /** Every budget up to 12, then a coarser grid up to tw(G+). */
    static List<Integer> budgetGrid(int twMax) {
        List<Integer> out = new ArrayList<>();
        for (int b = 2; b <= Math.min(12, twMax); b++) out.add(b);
        for (int b : new int[]{15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200})
            if (b < twMax) out.add(b);
        if (twMax > 12 && !out.contains(twMax)) out.add(twMax);
        return out;
    }
}
