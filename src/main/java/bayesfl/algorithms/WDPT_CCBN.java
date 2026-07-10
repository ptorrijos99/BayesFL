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
        this.cutPoints = cutPoints;
        this.options = Arrays.copyOf(options, options.length);
        this.globalClassMaps = globalClassMaps;
        this.noiseGenerator = noiseGenerator;
        this.paramDp = null;

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
        this(options, cutPoints, globalClassMaps, noiseGenerator);
        this.paramDp = paramDp;
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

            if (noiseGenerator != null) {
                privatizeTree(tree, modified, indices, originalData.classIndex(), classMap);

                // Refit the round-1 weights against the noisy tables so the shared
                // parameters are consistent with the tables used at prediction time
                try {
                    minimizer.run(algorithm.getObjectiveFunction(), tree.getParameters());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            trees.add(tree);
            functions.add(algorithm.getObjectiveFunction());
            classifiers.add(classifier);
            syntheticClassMaps.add(classMap);
        }

        return new WDPT(trees, classifiers, minimizers, combinations, syntheticClassMaps, functions, data.getNInstances());
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

        if (paramDp != null && paramDp.enabled()) {
            // Joint multi-tree DP-SGD refinement (Task 4's optimizer) over the
            // discriminative CLL data-loss gradient of every tree at once, so
            // the per-record clipping sensitivity is bounded across the whole
            // ensemble rather than per tree.
            int nTrees = oldTrees.size();
            double[][] treeParams = new double[nTrees][];
            SampleGradient[] grads = new SampleGradient[nTrees];
            double[] lambdas = new double[nTrees];
            for (int i = 0; i < nTrees; i++) {
                treeParams[i] = oldTrees.get(i).getParameters();
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
                oldTrees.get(i).copyParameters(treeParams[i]);
                newTrees.add(oldTrees.get(i));
            }
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

        // 2. Laplace-perturb the released counts and clip at zero
        double[] noisyClass = noiseGenerator.privatize(classCounts);
        double noisyN = 0.0;
        for (int c = 0; c < nc; c++) {
            noisyClass[c] = Math.max(0.0, noisyClass[c]);
            noisyN += noisyClass[c];
        }
        for (int u = 0; u < a; u++) {
            if (parentPos.containsKey(u)) continue;
            for (int v = 0; v < xyCounts[u].length; v++) {
                xyCounts[u][v] = noiseGenerator.privatize(xyCounts[u][v]);
                for (int c = 0; c < nc; c++) {
                    xyCounts[u][v][c] = Math.max(0.0, xyCounts[u][v][c]);
                }
            }
        }

        // 3. Overwrite the tree tables with probabilities from the noisy counts
        double[] treeClass = tree.getClassCounts();
        for (int c = 0; c < nc; c++) {
            treeClass[c] = mEsti(noisyClass[c], noisyN, nc);
        }

        int[] order = tree.getOrder();
        for (int pos = 0; pos < a; pos++) {
            int u = order[pos];
            wdBayesNode node = tree.wdBayesNode_[pos];
            int k = modified.attribute(u).numValues();

            if (parentPos.containsKey(u)) {
                // Deterministic reconstruction: all mass on the parent value
                // encoded in each synthetic-class label (noisy prior as denominator)
                int p = parentPos.get(u);
                for (Map.Entry<String, Integer> e : classMap.entrySet()) {
                    int y = e.getValue();
                    String[] parts = e.getKey().split("\\|\\|\\|");
                    int vStar = modified.attribute(u).indexOfValue(parts[p]);
                    if (vStar < 0) {
                        throw new IllegalStateException("Parent value '" + parts[p]
                                + "' not found in attribute " + modified.attribute(u).name());
                    }
                    for (int v = 0; v < k; v++) {
                        double count = (v == vStar) ? noisyClass[y] : 0.0;
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
