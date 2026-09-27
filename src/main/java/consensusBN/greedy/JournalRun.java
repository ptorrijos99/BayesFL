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
                case "mcbnc" -> runMcbnc(dags, w, row);
                case "greedyEa" -> runGreedyEa(dags, w, row);
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
        StructuralHillClimb s = new StructuralHillClimb(w);
        s.skelWeight = skelWeight;
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
