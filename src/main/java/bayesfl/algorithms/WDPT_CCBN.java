/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2024 Universidad de Castilla-La Mancha, España
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
 * Package containing algorithms related with federated Bayesian networks.
 */
package bayesfl.algorithms;

/**
 * Third-party imports.
 */
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import EBNC.wdBayes;
import objectiveFunction.ObjectiveFunction;
import optimize.Minimizer;
import optimize.StopConditions;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.util.*;

/**
 * Local application imports.
 */
import bayesfl.algorithms.dp.DPSGDConfig;
import bayesfl.algorithms.dp.DPSGDOptimizer;
import bayesfl.algorithms.dp.PerSampleCLLGradient;
import bayesfl.algorithms.dp.SampleGradient;
import bayesfl.data.Data;
import bayesfl.model.Model;
import bayesfl.model.TableCounts;
import bayesfl.model.WDPT;
import bayesfl.privacy.NumericNoiseGenerator;

import static org.albacete.simd.utils.Utils.*;


/**
 * A class representing a class-conditional Bayesian network algorithm.
 */
public class WDPT_CCBN implements LocalAlgorithm {

    /**
     * The maximum gradient norm.
     */
    private double maxGradientNorm = 0.00000000000000000000000000000001;

    /**
     * The name of the algorithm.
     */
    private String algorithmName = "WDBN_CCBN";

    /**
     * The name of the refinement method.
     */
    private String refinementName = "None";

    /**
     * The n of AnDE. 0 means Naive Bayes, 1 means A1DE, 2 means A2DE, etc.
     */
    private int nAnDE = 0;

    /**
     * The maximum number of iterations.
     */
    private int maxIterations;

    /**
     * The cut points of the discretization filter.
     */
    private final double[][] cutPoints;

    /**
     * The options to set the parameters of the algorithm.
     */
    private final String[] options;

    /**
     * Global class maps for synthetic classes.
     */
    private final List<Map<String, Integer>> globalClassMaps;

    /**
     * Optional client-side DP noise generator. When non-null, the raw count
     * tables of the round-1 build are Laplace-perturbed BEFORE being converted
     * to the probability tables that the fusion step shares (one-shot release,
     * formal epsilon-DP for the generative tables via the Laplace mechanism;
     * the later weight rounds never touch local counts again).
     */
    private final NumericNoiseGenerator noiseGenerator;

    /**
     * Optional discriminative-channel DP-SGD configuration for the per-round
     * parameter refinement ({@link #buildLocalModel(Model, Data)}). When
     * {@code null} or {@link DPSGDConfig#enabled()} is {@code false} (infinite
     * epsilon), the existing per-tree quasi-Newton minimizer loop is used
     * unchanged.
     */
    private DPSGDConfig paramDp;

    /**
     * Whether the tables are securely aggregated: the round-1 release is then
     * the raw counts plus a noise share, summed by the server.
     */
    private final boolean secureTableAggregation;

    /**
     * Constructor.
     *
     * @param options      The options to set the parameters of the algorithm.
     * @param cutPoints    The cut points of the discretization filter.
     */
    public WDPT_CCBN(String[] options, double[][] cutPoints, List<Map<String, Integer>> globalClassMaps) {
        this(options, cutPoints, globalClassMaps, null);
    }

    /**
     * Constructor with an optional DP noise generator for count-space privatization.
     *
     * @param options        The options to set the parameters of the algorithm.
     * @param cutPoints      The cut points of the discretization filter.
     * @param globalClassMaps Global class maps for synthetic classes.
     * @param noiseGenerator The DP noise generator applied to the raw counts (null disables DP).
     */
    public WDPT_CCBN(String[] options, double[][] cutPoints, List<Map<String, Integer>> globalClassMaps,
                     NumericNoiseGenerator noiseGenerator) {
        this(options, cutPoints, globalClassMaps, noiseGenerator, false);
    }

    /**
     * Shared constructor body.
     *
     * @param secureTableAggregation whether the tables are securely aggregated (noise shares allowed)
     */
    private WDPT_CCBN(String[] options, double[][] cutPoints, List<Map<String, Integer>> globalClassMaps,
                      NumericNoiseGenerator noiseGenerator, boolean secureTableAggregation) {
        this.secureTableAggregation = secureTableAggregation;
        this.cutPoints = cutPoints;
        this.options = Arrays.copyOf(options, options.length);
        this.globalClassMaps = globalClassMaps;
        this.noiseGenerator = noiseGenerator;
        this.paramDp = null;

        // A noise share is meaningless on its own: it only protects the tables when
        // the server sums the clients' counts (secure aggregation of the tables).
        // Otherwise the tables are fused by averaging probabilities, and shares
        // would not add up to the calibrated noise.
        if (noiseGenerator != null && noiseGenerator.isShare() && !secureTableAggregation) {
            throw new UnsupportedOperationException(
                "Secure-aggregation noise shares need secure aggregation of the tables: WDPT otherwise "
                + "fuses them by averaging probabilities, not by summing counts");
        }

        // Copy the options to avoid modifying the original array
        try {
            String structure = Utils.getOption("S", this.options);
            if (structure.startsWith("A") && structure.endsWith("DE")) {
                nAnDE = Integer.parseInt(structure.substring(1, structure.length() - 2));
            }
            // The internal wdBayes structure is always NB
            setInternalStructureToNB();
        } catch (Exception ignored) {}
    }

    /**
     * Constructor with an optional discriminative-channel DP-SGD configuration
     * for the per-round parameter refinement.
     *
     * @param options        The options to set the parameters of the algorithm.
     * @param cutPoints      The cut points of the discretization filter.
     * @param globalClassMaps Global class maps for synthetic classes.
     * @param noiseGenerator The DP noise generator applied to the raw counts (null disables count-space DP).
     * @param paramDp        The DP-SGD configuration for the parameter channel (null disables param-space DP).
     */
    public WDPT_CCBN(String[] options, double[][] cutPoints, List<Map<String, Integer>> globalClassMaps,
                     NumericNoiseGenerator noiseGenerator, DPSGDConfig paramDp) {
        this(options, cutPoints, globalClassMaps, noiseGenerator, paramDp, false);
    }

    /**
     * Constructor with secure aggregation of the tables (AnDEwP): in round 1
     * each client sends its raw counts plus a share of the noise, the server
     * sums them and writes the global tables, and the weights only take
     * private steps from round 2 on, with those global (public) tables as
     * their features.
     *
     * @param options                The options to set the parameters of the algorithm.
     * @param cutPoints              The cut points of the discretization filter.
     * @param globalClassMaps        Global class maps for synthetic classes.
     * @param noiseGenerator         The DP noise generator applied to the raw counts.
     * @param paramDp                The DP-SGD configuration for the parameter channel (null disables it).
     * @param secureTableAggregation Whether the tables are securely aggregated.
     */
    public WDPT_CCBN(String[] options, double[][] cutPoints, List<Map<String, Integer>> globalClassMaps,
                     NumericNoiseGenerator noiseGenerator, DPSGDConfig paramDp, boolean secureTableAggregation) {
        this(options, cutPoints, globalClassMaps, noiseGenerator, secureTableAggregation);
        this.paramDp = paramDp;

        // wCCBN weights multiply the log-probability tables, and those tables are
        // computed from every local record: without table noise one record shifts
        // every other record's gradient, so per-record clipping no longer bounds
        // the sensitivity by C.
        if (privateWeights() && noiseGenerator == null && isWeightedScheme()) {
            throw new IllegalArgumentException(
                "DP-SGD on wCCBN weights requires privatized count tables (-DP): their "
                + "log-probabilities are the per-record features of the weight gradient");
        }
    }

    /**
     * @return whether the round-1 tables are released as securely aggregated count shares.
     */
    private boolean sharesTables() {
        return secureTableAggregation && noiseGenerator != null;
    }

    /**
     * @return whether the released parameters are trained with DP-SGD.
     */
    private boolean privateWeights() {
        return paramDp != null && paramDp.enabled();
    }

    /**
     * @return whether the parameter-learning scheme ({@code -P}) is wCCBN.
     */
    private boolean isWeightedScheme() {
        try {
            return "wCCBN".equalsIgnoreCase(Utils.getOption("P", Arrays.copyOf(options, options.length)));
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Builds a local model using the provided data.
     *
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Data data) {
        if (noiseGenerator != null && hasActiveCutPoints(cutPoints)) {
            throw new UnsupportedOperationException(
                    "Count-space DP requires categorical data; on-the-fly numeric discretization "
                    + "is unsupported (its data-dependent cut points would be released unprotected)");
        }

        // Get the instances from the data
        Instances originalData = (Instances) data.getData();
        int nAttributes = originalData.numAttributes() - 1; // excluding class

        // Generate combinations of attributes (n-AnDE structure)
        List<int[]> combinations = generateCombinations(nAttributes, nAnDE);
        List<wdBayesParametersTree> trees = new ArrayList<>();
        List<AbstractClassifier> classifiers = new ArrayList<>();
        List<Minimizer> minimizers = new ArrayList<>();
        List<ObjectiveFunction> functions = new ArrayList<>();
        List<Map<String, Integer>> syntheticClassMaps = new ArrayList<>();
        List<TableCounts> countShares = new ArrayList<>();

        for (int i = 0; i < combinations.size(); i++) {
            int[] indices = combinations.get(i);
            Map<String, Integer> classMap = globalClassMaps.get(i);
            Instances modified = redefineClassAttribute(originalData, indices, classMap);
            //Instances completed = ensureAllClassValuesPresent(modified, classMap);

            // Build the classifier using the instances
            wdBayes algorithm = new wdBayes();
            try {
                algorithm.setOptions(Arrays.copyOf(options, options.length));
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Get the number of iterations from the algorithm
            this.maxIterations = algorithm.getM_MaxIterations();

            // Initialize the minimizer for the posterior parameter estimation
            Minimizer minimizer = new Minimizer();
            StopConditions sc = minimizer.getStopConditions();
            sc.setMaxGradientNorm(this.maxGradientNorm);
            sc.setMaxIterations(this.maxIterations);
            minimizers.add(minimizer);

            FilteredClassifier classifier = new FilteredClassifier();
            classifier.setClassifier(algorithm);

            // Set the discretization filter
            Filter filter;
            if (cutPoints != null) {
                filter = new Dummy();
                ((Dummy) filter).setCutPoints(cutPoints);
            } else {
                filter = new AllFilter(); // Bypass discretization
            }
            classifier.setFilter(filter);

            try {
                classifier.buildClassifier(modified);
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Save the tree-based storage and the objective function because
            // they are needed to build the local model from the global model
            wdBayesParametersTree tree = algorithm.getdParameters_();

            if (sharesTables()) {
                // Secure aggregation: release the raw counts plus this client's share
                // of the noise; the server sums them and writes the global tables.
                // The local tables are never released.
                countShares.add(rawCounts(modified, indices, originalData.classIndex())
                        .privatized(noiseGenerator, false));
            } else if (noiseGenerator != null) {
                privatizeTree(tree, modified, indices, originalData.classIndex(), classMap);

                // Refit the round-1 weights against the noisy tables so the shared
                // parameters are consistent with the tables used at prediction time.
                // Under DP-SGD the weights are released, so the private round below
                // learns them instead of this non-private refit.
                if (!privateWeights()) {
                    try {
                        minimizer.run(algorithm.getObjectiveFunction(), tree.getParameters());
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }

            // Under DP-SGD, discard what buildClassifier's optimizer learned from the
            // raw records and restart from the scheme's constant initialization, which
            // is data-independent: the private round is then the only thing that moves
            // the parameters this client releases in round 1.
            if (privateWeights()) {
                tree.initializeParameters(algorithm.getM_WeightingInitialization());
            }

            trees.add(tree);
            functions.add(algorithm.getObjectiveFunction());
            classifiers.add(classifier);
            syntheticClassMaps.add(classMap);
        }

        // With securely aggregated tables the features of round 1 would be the local,
        // unreleased tables: the weights stay at their public start until the global
        // tables arrive, and take their first private step in round 2.
        if (privateWeights() && !sharesTables()) {
            runPrivateRound(trees, classifiers, data);
        }

        WDPT model = new WDPT(trees, classifiers, minimizers, combinations, syntheticClassMaps, functions, data.getNInstances());
        if (sharesTables()) model.setCountShares(countShares);
        return model;
    }

    /**
     * Builds a local model using the provided data and existing local model.
     *
     * @param localModel The existing local model.
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Model localModel, Data data) {
        // If there is no local model, build a new one
        if (localModel == null) {
            return this.buildLocalModel(data);
        }

        // Retrieve trees, classifiers, and objective functions from the previous model
        WDPT previous = (WDPT) localModel;
        List<wdBayesParametersTree> oldTrees = previous.getTrees();
        List<AbstractClassifier> classifiers = previous.getClassifiers();
        List<Minimizer> minimizers = previous.getMinimizers();
        List<ObjectiveFunction> functions = previous.getFunctions();
        List<int[]> combinations = previous.getCombinations();
        List<Map<String, Integer>> classMaps = previous.getSyntheticClassMaps();

        List<wdBayesParametersTree> newTrees = new ArrayList<>();

        if (privateWeights()) {
            runPrivateRound(oldTrees, classifiers, data);
            newTrees.addAll(oldTrees);
        } else {
            // Optimize each tree individually using its objective function
            for (int i = 0; i < oldTrees.size(); i++) {
                wdBayesParametersTree tree = oldTrees.get(i);
                double[] parameters = tree.getParameters();
                try {
                    minimizers.get(i).run(functions.get(i), parameters);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                // Reuse the tree reference, since it has been updated internally
                newTrees.add(tree);
            }
        }

        // Return a new model instance with updated parameters
        return new WDPT(newTrees, classifiers, minimizers, combinations, classMaps, functions, data.getNInstances());
    }

    /**
     * Runs one round of joint multi-tree DP-SGD over the discriminative CLL
     * data-loss gradient of every tree at once, so the per-record clipping
     * sensitivity is bounded across the whole ensemble rather than per tree.
     * Every round a client releases, the first included, goes through here.
     *
     * @param trees       The parameter trees, updated in place.
     * @param classifiers The classifiers the trees belong to, in the same order.
     * @param data        The local data.
     */
    private void runPrivateRound(List<wdBayesParametersTree> trees, List<AbstractClassifier> classifiers, Data data) {
        int nTrees = trees.size();
        double[][] treeParams = new double[nTrees][];
        SampleGradient[] grads = new SampleGradient[nTrees];
        double[] lambdas = new double[nTrees];
        for (int i = 0; i < nTrees; i++) {
            treeParams[i] = trees.get(i).getParameters();
            wdBayes alg = (wdBayes) ((FilteredClassifier) classifiers.get(i)).getClassifier();
            grads[i] = new PerSampleCLLGradient(alg);
            lambdas[i] = alg.getRegularization() ? alg.getLambda() : 0.0;
        }
        new DPSGDOptimizer(paramDp).runRound(treeParams, grads, lambdas, data.getNInstances());

        // Re-sync each tree's trie with the final parameters: the gradient
        // calls above leave it in a stale "probe" state (PerSampleCLLGradient
        // re-syncs the trie from whichever params it was last called with),
        // so this write-back must run before the tree is reused for inference.
        for (int i = 0; i < nTrees; i++) {
            trees.get(i).copyParameters(treeParams[i]);
        }
    }

    /**
     * Refines the existing local model using the provided data. In this case, the model is not refined.
     *
     * @param oldModel The existing local model.
     * @param localModel The current local model.
     * @param data The data to refine the model with.
     * @return The refined local model.
     */
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return localModel;
    }

    /**
     * Ensures that the structure option (-S) is set to "NB" in the options array.
     * <p>
     * If the -S flag is already present, replaces its associated value with "NB".
     * If not present, inserts "-S" and "NB" into the first pair of empty positions.
     * </p>
     */
    private void setInternalStructureToNB() {
        int pos = -1;
        for (int i = 0; i < this.options.length; i++) {
            if ("-S".equals(this.options[i])) {
                pos = i;
                break;
            }
        }

        if (pos != -1 && pos + 1 < this.options.length) {
            this.options[pos + 1] = "NB";
        } else {
            for (int i = 0; i < this.options.length - 1; i++) {
                if (this.options[i].isEmpty() && this.options[i + 1].isEmpty()) {
                    this.options[i] = "-S";
                    this.options[i + 1] = "NB";
                    break;
                }
            }
        }
    }


    /**
     * Replaces the tree's probability tables with tables derived from
     * Laplace-perturbed raw counts (the formal DP release).
     * <p>
     * The released statistics per SPnDE are the synthetic-class prior counts
     * and the conditional counts of NON-parent attributes (per-record L1
     * sensitivity C(a,n)*(1+a-n), Proposition 1). Parent-attribute tables are
     * deterministic given the synthetic class and are reconstructed from the
     * noisy prior (post-processing). Probability conversion replicates
     * wdBayesParametersTree.countsToProbability: linear-scale m-estimates for
     * the class prior, log-scale m-estimates clamped at 1e-75 for conditionals.
     */
    private void privatizeTree(wdBayesParametersTree tree, Instances modified, int[] parentOriginalIdx,
                               int origClassIdx, Map<String, Integer> classMap) {
        // 1. Recount, 2. Laplace-perturb and clip at zero, 3. write the tables.
        TableCounts noisy = rawCounts(modified, parentOriginalIdx, origClassIdx).privatized(noiseGenerator, true);
        writeTablesFromCounts(tree, noisy, classMap);
    }

    /**
     * Recounts the released sufficient statistics of one SPnDE from the data it
     * was built on (the tree already holds probabilities): synthetic-class prior
     * counts and the conditional counts of the non-parent attributes.
     *
     * @param modified          the tree's relabelled data
     * @param parentOriginalIdx the parent attributes of the combination, in original indices
     * @param origClassIdx      the class index in the original data
     * @return the raw counts
     */
    private static TableCounts rawCounts(Instances modified, int[] parentOriginalIdx, int origClassIdx) {
        int nc = modified.classAttribute().numValues();
        int a = modified.numAttributes() - 1; // predictive attributes; synthetic class is last

        // Parent positions in the relabelled data and their index inside the combination
        Map<Integer, Integer> parentPos = new HashMap<>();
        for (int p = 0; p < parentOriginalIdx.length; p++) {
            int mod = parentOriginalIdx[p] < origClassIdx ? parentOriginalIdx[p] : parentOriginalIdx[p] - 1;
            parentPos.put(mod, p);
        }

        // 1. Recount raw sufficient statistics (the tree already holds probabilities)
        double[] classCounts = new double[nc];
        double[][][] xyCounts = new double[a][][];
        for (int u = 0; u < a; u++) {
            xyCounts[u] = new double[modified.attribute(u).numValues()][nc];
        }
        for (int r = 0; r < modified.numInstances(); r++) {
            weka.core.Instance inst = modified.instance(r);
            int y = (int) inst.classValue();
            classCounts[y]++;
            for (int u = 0; u < a; u++) {
                if (parentPos.containsKey(u)) continue; // not released
                xyCounts[u][(int) inst.value(u)][y]++;
            }
        }

        return new TableCounts(modified, parentPos, classCounts, xyCounts);
    }

    /**
     * Overwrites a tree's probability tables with tables derived from (noisy)
     * counts. Probability conversion replicates
     * wdBayesParametersTree.countsToProbability: linear-scale m-estimates for the
     * class prior, log-scale m-estimates clamped at 1e-75 for conditionals; parent
     * tables are reconstructed deterministically from the prior. The server uses
     * it too, to write the global tables from the securely aggregated counts.
     *
     * @param tree     the tree to overwrite
     * @param counts   the counts, already clamped at zero
     * @param classMap the synthetic class map of the tree
     */
    public static void writeTablesFromCounts(wdBayesParametersTree tree, TableCounts counts,
                                             Map<String, Integer> classMap) {
        Instances header = counts.header();
        Map<Integer, Integer> parentPos = counts.parentPos();
        double[] noisyClass = counts.cls();
        double[][][] xyCounts = counts.xy();
        int nc = header.classAttribute().numValues();
        int a = header.numAttributes() - 1;
        double noisyN = 0.0;
        for (double c : noisyClass) noisyN += c;

        // 3. Overwrite the tree tables with probabilities from the noisy counts
        double[] treeClass = tree.getClassCounts();
        for (int c = 0; c < nc; c++) {
            treeClass[c] = mEsti(noisyClass[c], noisyN, nc);
        }

        int[] order = tree.getOrder();
        for (int pos = 0; pos < a; pos++) {
            int u = order[pos];
            wdBayesNode node = tree.wdBayesNode_[pos];
            int k = header.attribute(u).numValues();

            if (parentPos.containsKey(u)) {
                // Deterministic reconstruction: all mass on the parent value
                // encoded in each synthetic-class label (noisy prior as denominator)
                int p = parentPos.get(u);
                for (Map.Entry<String, Integer> e : classMap.entrySet()) {
                    int y = e.getValue();
                    String[] parts = e.getKey().split("\\|\\|\\|");
                    int vStar = header.attribute(u).indexOfValue(parts[p]);
                    if (vStar < 0 && !"?".equals(parts[p])) {
                        throw new IllegalStateException("Parent value '" + parts[p]
                                + "' not found in attribute " + header.attribute(u).name());
                    }
                    // parts[p] == "?" (Weka's missing marker, vStar < 0): no observed
                    // parent value for this synthetic class -> spread the mass uniformly
                    // (deterministic post-processing of the noisy prior, no leak).
                    for (int v = 0; v < k; v++) {
                        double count = vStar < 0 ? noisyClass[y] / k
                                : ((v == vStar) ? noisyClass[y] : 0.0);
                        node.setXYCount(v, y, Math.log(Math.max(mEsti(count, noisyClass[y], k), 1e-75)));
                    }
                }
            } else {
                for (int y = 0; y < nc; y++) {
                    double denom = 0.0;
                    for (int v = 0; v < k; v++) denom += xyCounts[u][v][y];
                    for (int v = 0; v < k; v++) {
                        node.setXYCount(v, y, Math.log(Math.max(mEsti(xyCounts[u][v][y], denom, k), 1e-75)));
                    }
                }
            }
        }
    }

    /** Replicates EBNC SUtils.MEsti with m = 1. */
    private static double mEsti(double freq, double total, double numValues) {
        return (freq + 1.0 / numValues) / (total + 1.0);
    }

    /**
     * Retrieves the name of the algorithm.
     *
     * @return The name of the algorithm.
     */
    public String getAlgorithmName() {
        return this.algorithmName;
    }

    /** 
     * Retrieves the name of the refinement.
     *
     * @return The name of the refinement.
     */
    public String getRefinementName() {
        return this.refinementName;
    }
}
