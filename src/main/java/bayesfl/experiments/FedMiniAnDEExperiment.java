/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2026 Universidad de Castilla-La Mancha, España
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */
/**
 *    FedMiniAnDEExperiment.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *
 *    Memory-efficient two-round federated MiniAnDE experiment with multi-γ
 *    evaluation. Clients are processed sequentially and fused incrementally
 *    to avoid holding K models in memory simultaneously.
 *
 *      Round 1 (Structure): Each client builds trees locally, extracts mSPnDE
 *              structures, incrementally fused via union.
 *      Round 2 (Parameters): Using the fused structure, each client builds
 *              count tables, incrementally accumulated and normalized.
 *
 *    The addNB (γ) parameter is swept post-hoc at evaluation time: the model
 *    is trained once, then evaluated with each γ value, saving to separate
 *    files. This avoids redundant training runs.
 *
 * @author Pablo Torrijos Arenas
 */

package bayesfl.experiments;

import bayesfl.Client;
import bayesfl.Server;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.algorithms.mAnDETree_LocalParams;
import bayesfl.algorithms.mAnDETree_mAnDE;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Weka_Instances;
import bayesfl.fusion.Fusion;
import bayesfl.fusion.Bins_Fusion;
import bayesfl.algorithms.Bins_Unsupervised;
import bayesfl.model.Bins;
import bayesfl.model.mAnDETree;
import bayesfl.privacy.NumericNoiseGenerator;
import bayesfl.privacy.Laplace_Noise;
import org.albacete.simd.mAnDE.mSPnDE;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.utils.ExperimentUtils.readParametersFromArgs;

public class FedMiniAnDEExperiment {
    public static String PATH = "./";

    public static final double[] GAMMA_VALUES = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0};

    public static void main(String[] args) {
        String nodeName = "localhost";
        String folder = "Discretas";
        String bbdd = "Car_Evaluation";
        int nClients = 1;
        int seed = 42;
        int folds = 2;
        int n = 1;
        int nTrees = 100;
        double bagSize = 100.0;
        String ensemble = "Bagging";

        double epsilon = 0.0;
        String discMode = "FedF10";
        double alpha = -1.0;

        NumericNoiseGenerator noiseGenerator = null;

        if (args.length > 0) {
            if (args.length >= 3) {
                nodeName = args[2];
            }
            args = readParametersFromArgs(args);

            folder = args[0];
            bbdd = args[1];
            nClients = Integer.parseInt(args[2]);
            seed = Integer.parseInt(args[3]);
            folds = Integer.parseInt(args[4]);
            n = Integer.parseInt(args[5]);
            nTrees = Integer.parseInt(args[6]);
            bagSize = Double.parseDouble(args[7]);
            ensemble = args[8];
            // args[9] = addNB (ignored, swept post-hoc via GAMMA_VALUES)

            if (args.length > 10) {
                epsilon = Double.parseDouble(args[10]);
                discMode = args[11];
                alpha = Double.parseDouble(args[12]);
            }
        }

        // Optional CG (conditional-Gaussian) trailing parameters (backward
        // compatible with 13-field lines): args[13]=continuous, args[14]=cgPriorVarDof.
        boolean continuous = false;
        double cgPriorVarDof = 3.0;
        boolean cgNBDiscretized = false;
        if (args.length > 13) continuous = args[13].equalsIgnoreCase("true") || args[13].equals("1");
        if (args.length > 14) cgPriorVarDof = Double.parseDouble(args[14]);
        // args[15] = cgNBDiscretized: build the blended NB on discretized (categorical)
        // data instead of a full-dimensional Gaussian NB, so smoothing (γ > 0) is usable
        // in continuous mode.
        if (args.length > 15) cgNBDiscretized = args[15].equalsIgnoreCase("true") || args[15].equals("1");
        // args[16] = buildOnly: write only the Round-1 Build stats and skip the Fusion
        // evaluation and Round 2. At K=1 both are redundant with Build.
        boolean buildOnly = false;
        if (args.length > 16) buildOnly = args[16].equalsIgnoreCase("true") || args[16].equals("1");

        if (epsilon > 0) {
            noiseGenerator = new Laplace_Noise(epsilon, 2.0);
        }

        experimentFedMiniAnDE(folder, bbdd, nClients, seed, folds, n, nTrees, bagSize, ensemble,
                epsilon, discMode, alpha, noiseGenerator, nodeName, continuous, cgPriorVarDof, cgNBDiscretized,
                buildOnly);
    }

    public static void experimentFedMiniAnDE(String folder, String bbdd, int nClients, int seed, int nFolds,
                                             int n, int nTrees, double bagSize, String ensemble,
                                             double epsilon, String discMode, double alpha,
                                             NumericNoiseGenerator noiseGenerator, String nodeName) {
        experimentFedMiniAnDE(folder, bbdd, nClients, seed, nFolds, n, nTrees, bagSize, ensemble,
                epsilon, discMode, alpha, noiseGenerator, nodeName, false, 3.0, false);
    }

    public static void experimentFedMiniAnDE(String folder, String bbdd, int nClients, int seed, int nFolds,
                                             int n, int nTrees, double bagSize, String ensemble,
                                             double epsilon, String discMode, double alpha,
                                             NumericNoiseGenerator noiseGenerator, String nodeName,
                                             boolean continuous, double cgPriorVarDof) {
        experimentFedMiniAnDE(folder, bbdd, nClients, seed, nFolds, n, nTrees, bagSize, ensemble,
                epsilon, discMode, alpha, noiseGenerator, nodeName, continuous, cgPriorVarDof, false);
    }

    public static void experimentFedMiniAnDE(String folder, String bbdd, int nClients, int seed, int nFolds,
                                             int n, int nTrees, double bagSize, String ensemble,
                                             double epsilon, String discMode, double alpha,
                                             NumericNoiseGenerator noiseGenerator, String nodeName,
                                             boolean continuous, double cgPriorVarDof, boolean cgNBDiscretized) {
        experimentFedMiniAnDE(folder, bbdd, nClients, seed, nFolds, n, nTrees, bagSize, ensemble,
                epsilon, discMode, alpha, noiseGenerator, nodeName, continuous, cgPriorVarDof, cgNBDiscretized,
                false);
    }

    public static void experimentFedMiniAnDE(String folder, String bbdd, int nClients, int seed, int nFolds,
                                             int n, int nTrees, double bagSize, String ensemble,
                                             double epsilon, String discMode, double alpha,
                                             NumericNoiseGenerator noiseGenerator, String nodeName,
                                             boolean continuous, double cgPriorVarDof, boolean cgNBDiscretized,
                                             boolean buildOnly) {

        // CG (conditional-Gaussian) is centralized-only: the federated parameter
        // fusion (Round 2) is not implemented for CG SPnDEs, so force local-only.
        // CG also requires NON-discretized children (discMode should be "None"):
        // mAnDE discretizes the super-parent internally and keeps children numeric.
        boolean localOnly = epsilon < 0 || continuous;
        String epTag = localOnly ? "" : (epsilon > 0 ? "_ep" + epsilon : "");
        String variantTag = discMode + epTag + (alpha > 0 ? "_a" + alpha : "");

        // "nbd" marks a categorical (discretized) blended NB, distinguishing these
        // runs from the default Gaussian-NB CG runs in the results.
        String cgTag = continuous ? "-CGd" + cgPriorVarDof + (cgNBDiscretized ? "nbd" : "") : "";
        String r1AlgoTag = localOnly
                ? "mA" + n + "DE" + cgTag + "-Local_" + variantTag
                : "mA" + n + "DE" + cgTag + "-FedAnDE_" + variantTag;
        String r2AlgoTag = "mA" + n + "DE" + cgTag + "-FedParams_" + variantTag;

        // Operation template: algorithm,seed,nTrees,bagSize,ensemble,{γ}
        // The addNB (γ) field is replaced per-γ in mAnDETree.saveStats
        String opBase = r1AlgoTag + "," + seed + "," + nTrees + "," + bagSize + "," + ensemble + ",";
        String r2OpBase = r2AlgoTag + "," + seed + "," + nTrees + "," + bagSize + "," + ensemble + ",";

        // Use the first γ (0.0) as the canonical addNB for operation strings
        String operation = opBase + "0.0";
        String round2Op = r2OpBase + "0.0";

        String bbddPath = PATH + "res/classification/" + folder + "/" + bbdd + ".arff";
        Instances[][][] splits = divide(bbdd, bbddPath, nFolds, nClients, seed, alpha);
        int actualFolds = splits.length;

        // Resume: a fold counts as done once every output file holds it for all
        // clients. Rows of unfinished folds are dropped, so a killed job restarts
        // from its first missing fold without duplicating rows.
        List<String> outputs = outputFiles(bbdd, opBase, r2OpBase, nClients, localOnly, buildOnly);
        Set<Integer> doneFolds = completedFolds(outputs, nClients);
        if (doneFolds.size() >= actualFolds) {
            System.out.println("Experiment already complete (all γ files present). Skipping "
                    + r1AlgoTag + " bbdd=" + bbdd + " K=" + nClients + " seed=" + seed);
            return;
        }
        for (String f : outputs) {
            keepOnlyFolds(f, doneFolds);
        }
        System.out.println("Resuming " + r1AlgoTag + " bbdd=" + bbdd + " seed=" + seed + ": "
                + doneFolds.size() + "/" + actualFolds + " folds already done.");

        for (int cv = 0; cv < actualFolds; cv++) {
            // Detach the fold from `splits`: the discretized copies written back into
            // it would otherwise stay reachable for the whole run, one dataset per fold.
            Instances[][] foldSplits = splits[cv];
            splits[cv] = null;
            if (doneFolds.contains(cv)) continue;

            // ── Phase 0: Discretization ─────────────────────────────────
            System.out.println("\n=== ROUND 0: Discretization (cv=" + cv + ", mode=" + discMode + ") ===\n");

            double[][] cutPoints = null;

            if (discMode.equals("None")) {
                System.out.println("Skipping discretization (data assumed already discrete).");
            } else if (discMode.startsWith("Central")) {
                applyCentralDiscretization(foldSplits, nClients, discMode);
            } else {
                cutPoints = federatedDiscretization(foldSplits, nClients, bbdd, cv);
            }

            if (cutPoints != null) {
                try {
                    bayesfl.algorithms.Dummy explicitFilter = new bayesfl.algorithms.Dummy();
                    explicitFilter.setCutPoints(cutPoints);
                    explicitFilter.setInputFormat(foldSplits[0][0]);
                    for (int i = 0; i < nClients; i++) {
                        foldSplits[i][0] = weka.filters.Filter.useFilter(foldSplits[i][0], explicitFilter);
                        foldSplits[i][1] = weka.filters.Filter.useFilter(foldSplits[i][1], explicitFilter);
                    }
                    System.out.println("Dataset explicitly discretized using global cut points.");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            // ── Phase 1: Structure learning (sequential + incremental) ──
            System.out.println("\n=== ROUND 1: Structure Learning (cv=" + cv + ") ===\n");

            ConcurrentHashMap<Object, mSPnDE> fusedStructure = new ConcurrentHashMap<>();
            mAnDETree_mAnDE algorithmRef = null;

            for (int i = 0; i < nClients; i++) {
                Weka_Instances data = new Weka_Instances(
                        bbdd + "," + i + "," + cv + "," + nodeName,
                        foldSplits[i][0], foldSplits[i][1]);

                mAnDETree_mAnDE alg = new mAnDETree_mAnDE(n, nTrees, bagSize, ensemble, 0.0, cutPoints);
                alg.setAddNBValues(GAMMA_VALUES);
                alg.setContinuous(continuous);
                alg.setCgPriorVarDof(cgPriorVarDof);
                alg.setCgNBDiscretized(cgNBDiscretized);

                long start = System.currentTimeMillis();
                mAnDETree localModel = (mAnDETree) alg.buildLocalModel(null, data);
                double buildTime = (System.currentTimeMillis() - start) / 1000.0;

                // Incrementally fuse structure BEFORE saveStats (which modifies the model)
                ConcurrentHashMap<Object, mSPnDE> localSPnDEs = localModel.getModel();
                for (Object key : localSPnDEs.keySet()) {
                    if (fusedStructure.containsKey(key)) {
                        fusedStructure.get(key).moreChildren(localSPnDEs.get(key).getChildren());
                    } else {
                        fusedStructure.put(key, localSPnDEs.get(key).copyDeep());
                    }
                }

                // Save Build stats (multi-γ, each γ → separate file)
                localModel.saveStats(operation, "Client/Build", PATH, nClients, i, data, 1, buildTime);

                if (algorithmRef == null) algorithmRef = alg;
                System.out.println("  Client " + i + "/" + nClients + " structure done (" + localSPnDEs.size() + " SPODEs)");
            }

            if (buildOnly) continue;

            mAnDETree globalStructure = new mAnDETree(fusedStructure, algorithmRef);
            System.out.println("Fused structure has " + fusedStructure.size() + " SPODEs (from " + nClients + " clients)");

            // Save Fusion stats: evaluate global structure on each client's data
            for (int i = 0; i < nClients; i++) {
                Weka_Instances data = new Weka_Instances(
                        bbdd + "," + i + "," + cv + "," + nodeName,
                        foldSplits[i][0], foldSplits[i][1]);
                globalStructure.saveStats(operation, "Client/Fusion", PATH, nClients, i, data, 1, 0);
            }

            if (localOnly) continue;

            // ── Phase 2: Parameter federation (sequential + incremental) ─
            List<int[]> combinations = globalStructure.toCombinations();
            if (combinations.isEmpty()) {
                System.out.println("WARNING: No SPODEs found. Skipping parameter federation for cv=" + cv);
                continue;
            }

            System.out.println("\n=== ROUND 2: Parameter Federation (cv=" + cv
                    + (noiseGenerator != null ? ", DP=" + noiseGenerator.getClass().getSimpleName() : "")
                    + ") ===\n");

            // Memory-efficient accumulation: the FIRST client builds the
            // accumulator (one full count-table model); every SUBSEQUENT client
            // counts its data DIRECTLY into the accumulator, one SPODE at a time,
            // via accumulateInto(). This avoids ever holding a full second copy
            // of the (large) global model — peak Round-2 memory drops from ~2x a
            // full union model to ~1x + one SPODE per worker thread, which is
            // what lets high-K A2DE runs on high-dimensional data fit in memory.
            // (copyDeep() only copies structure, not counts, so the first
            // client's model is reused directly as the accumulator. Per-client
            // Round 2 Build stats are skipped; only Fusion stats matter.)
            ConcurrentHashMap<Object, mSPnDE> accumulatedCounts = null;

            for (int i = 0; i < nClients; i++) {
                Weka_Instances data = new Weka_Instances(
                        bbdd + "," + i + "," + cv + "," + nodeName,
                        foldSplits[i][0], foldSplits[i][1]);

                mAnDETree_LocalParams paramAlg = new mAnDETree_LocalParams(globalStructure, null, noiseGenerator);

                long start = System.currentTimeMillis();
                if (accumulatedCounts == null) {
                    mAnDETree clientParamModel = (mAnDETree) paramAlg.buildLocalModel(data);
                    accumulatedCounts = clientParamModel.getModel();
                } else {
                    paramAlg.accumulateInto(accumulatedCounts, data);
                }
                double buildTime = (System.currentTimeMillis() - start) / 1000.0;

                System.out.println("  Client " + i + "/" + nClients + " params done (" + buildTime + "s)");
            }

            // Normalize accumulated counts → probability distributions
            accumulatedCounts.values().parallelStream().forEach(mSPnDE::normalizeCounts);
            mAnDETree globalParams = new mAnDETree(accumulatedCounts, algorithmRef);
            System.out.println("Parameter federation complete. Global model has " + accumulatedCounts.size() + " SPODEs.");

            // Save Fusion stats: evaluate global param model on each client
            for (int i = 0; i < nClients; i++) {
                Weka_Instances data = new Weka_Instances(
                        bbdd + "," + i + "," + cv + "," + nodeName,
                        foldSplits[i][0], foldSplits[i][1]);
                globalParams.saveStats(round2Op, "Client/Fusion", PATH, nClients, i, data, 1, 0);
            }
        }
    }

    // ── Resume support ──────────────────────────────────────────────────

    /** Every CSV this run writes, one per γ and role. */
    private static List<String> outputFiles(String bbdd, String opBase, String r2OpBase, int nClients,
                                            boolean localOnly, boolean buildOnly) {
        List<String> files = new ArrayList<>();
        for (double gamma : GAMMA_VALUES) {
            files.add(resultFile("Client/Build", bbdd, opBase + gamma, nClients));
            if (buildOnly) continue;
            files.add(resultFile("Client/Fusion", bbdd, opBase + gamma, nClients));
            if (!localOnly) {
                files.add(resultFile("Client/Fusion", bbdd, r2OpBase + gamma, nClients));
            }
        }
        return files;
    }

    private static String resultFile(String role, String bbdd, String op, int nClients) {
        return PATH + "results/" + role + "/" + bbdd + "_" + op + "_" + nClients + ".csv";
    }

    /** Folds present for all clients in every file (columns: bbdd,id,cv,...). */
    private static Set<Integer> completedFolds(List<String> files, int nClients) {
        Set<Integer> done = null;
        for (String path : files) {
            Map<Integer, Set<Integer>> clientsPerFold = new HashMap<>();
            java.io.File f = new java.io.File(path);
            if (f.isFile()) {
                try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(f))) {
                    String header = br.readLine();
                    int nCols = header == null ? 0 : header.split(",", -1).length;
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] cols = line.split(",", -1);
                        if (cols.length != nCols) continue; // truncated last line of a killed job
                        try {
                            clientsPerFold.computeIfAbsent(Integer.parseInt(cols[2].trim()), k -> new HashSet<>())
                                    .add(Integer.parseInt(cols[1].trim()));
                        } catch (NumberFormatException ignored) {
                            // malformed row
                        }
                    }
                } catch (java.io.IOException e) {
                    return new HashSet<>();
                }
            }
            Set<Integer> fileDone = new HashSet<>();
            for (Map.Entry<Integer, Set<Integer>> e : clientsPerFold.entrySet()) {
                if (e.getValue().size() >= nClients) fileDone.add(e.getKey());
            }
            if (done == null) done = fileDone;
            else done.retainAll(fileDone);
        }
        return done == null ? new HashSet<>() : done;
    }

    /** Rewrites a result file keeping the header and the rows of the given folds. */
    private static void keepOnlyFolds(String path, Set<Integer> folds) {
        java.io.File f = new java.io.File(path);
        if (!f.isFile()) return;
        List<String> kept = new ArrayList<>();
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(f))) {
            String header = br.readLine();
            if (header == null) return;
            kept.add(header);
            int nCols = header.split(",", -1).length;
            String line;
            while ((line = br.readLine()) != null) {
                String[] cols = line.split(",", -1);
                if (cols.length != nCols) continue;
                try {
                    if (folds.contains(Integer.parseInt(cols[2].trim()))) kept.add(line);
                } catch (NumberFormatException ignored) {
                    // drop malformed rows
                }
            }
        } catch (java.io.IOException e) {
            System.err.println("WARNING: could not read " + f + " for resume: " + e.getMessage());
            return;
        }
        try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(f, false))) {
            for (String l : kept) pw.println(l);
        } catch (java.io.IOException e) {
            System.err.println("WARNING: could not rewrite " + f + " for resume: " + e.getMessage());
        }
    }

    // ── Discretization helpers (unchanged) ──────────────────────────────

    private static double[][] federatedDiscretization(Instances[][] foldSplits, int nClients, String bbdd, int cv) {
        ArrayList<Client> discClients = new ArrayList<>();
        String[] discretizerOptions = new String[] { "-F", "-B", "10" };

        for (int i = 0; i < nClients; i++) {
            Instances train = foldSplits[i][0];
            Instances test = foldSplits[i][1];
            Weka_Instances data = new Weka_Instances((bbdd + "," + i + "," + cv), train, test);

            LocalAlgorithm discAlgorithm = new Bins_Unsupervised(discretizerOptions);
            Fusion fusionClient = new bayesfl.fusion.FusionPosition(-1);
            Client client = new Client(fusionClient, discAlgorithm, data);

            client.setStats(false, false, PATH);
            client.setExperimentName("Disc_FedMiniAnDE");
            client.setID(i);
            discClients.add(client);
        }

        Fusion discFusionServer = new Bins_Fusion();
        Server discServer = new Server(discFusionServer, new NoneConvergence(), discClients);
        discServer.setStats(false, PATH);
        discServer.setExperimentName("Disc_FedMiniAnDE");
        discServer.setnIterations(1);
        discServer.run();

        Bins globalBins = (Bins) discServer.getGlobalModel();
        System.out.println("Federated discretization complete. Cut points generated.");
        return globalBins.getModel();
    }

    private static void applyCentralDiscretization(Instances[][] foldSplits, int nClients, String discMode) {
        try {
            Instances globalTrain = new Instances(foldSplits[0][0], 0);
            for (int i = 0; i < nClients; i++) {
                Instances clientTrain = foldSplits[i][0];
                for (int j = 0; j < clientTrain.numInstances(); j++) {
                    globalTrain.add(clientTrain.instance(j));
                }
            }

            weka.filters.Filter centralFilter;
            switch (discMode) {
                case "CentralKono": {
                    weka.filters.supervised.attribute.Discretize d = new weka.filters.supervised.attribute.Discretize();
                    d.setUseKononenko(true);
                    centralFilter = d;
                    break;
                }
                case "CentralF10": {
                    weka.filters.unsupervised.attribute.Discretize d = new weka.filters.unsupervised.attribute.Discretize();
                    d.setOptions(new String[] { "-F", "-B", "10" });
                    centralFilter = d;
                    break;
                }
                case "CentralW10": {
                    weka.filters.unsupervised.attribute.Discretize d = new weka.filters.unsupervised.attribute.Discretize();
                    d.setOptions(new String[] { "-B", "10" });
                    centralFilter = d;
                    break;
                }
                case "CentralMDL":
                default:
                    centralFilter = new weka.filters.supervised.attribute.Discretize();
                    break;
            }

            centralFilter.setInputFormat(globalTrain);
            weka.filters.Filter.useFilter(globalTrain, centralFilter);

            for (int i = 0; i < nClients; i++) {
                foldSplits[i][0] = weka.filters.Filter.useFilter(foldSplits[i][0], centralFilter);
                foldSplits[i][1] = weka.filters.Filter.useFilter(foldSplits[i][1], centralFilter);
            }
            System.out.println("Central discretization (" + discMode + ") applied to all client splits.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
