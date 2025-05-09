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
 * Package containing fusion methods models with federated Bayesian networks.
 */
package bayesfl.fusion;

import DataStructure.wdBayesNode;
/**
 * Third-party imports.
 */
import DataStructure.wdBayesParametersTree;
import objectiveFunction.ObjectiveFunction;
import optimize.Minimizer;
import org.apache.commons.math3.util.MathArrays;

/**
 * Local application imports.
 */
import bayesfl.model.Model;
import bayesfl.model.WDPT;
import weka.classifiers.AbstractClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A class representing a fusion method for class-conditional Bayesian networks in the server.
 */
public class WDPT_Fusion_Server implements Fusion {

    /**
     * Whether to fuse the parameters or not.
     */
    private boolean fuseParameters;

    /**
     * Whether to fuse the probabilities or not.
     */
    private boolean fuseProbabilities;

    /**
     * Minimum allowable probability value used to prevent underflow and log(0)
     * during normalization and log-space conversion. Ensures numerical stability
     * when probabilities are extremely small or zero.
     */
    private static final double MIN_PROBABILITY = 1e-75;

    /** Constructor.
     * 
     * @param fuseParameters Whether to fuse the parameters or not.
     * @param fuseProbabilities Whether to fuse the probabilities or not.
     */
    public WDPT_Fusion_Server(boolean fuseParameters, boolean fuseProbabilities) {
        this.fuseParameters = fuseParameters;
        this.fuseProbabilities = fuseProbabilities;
    }

    /**
     * Perform the fusion of two models.
     * 
     * @param model1 The first model to fuse.
     * @param model2 The second model to fuse.
     * @return The global model fused.
     */
    public Model fusion(Model model1, Model model2) {
        return fusion(new Model[]{model1, model2});
    }

    /**
     * Fuses an array of local models into a single global model.
     * The fusion can be based on either the model parameters, the probability distributions,
     * or both, depending on the configuration flags.
     * <p>
     * Assumes all input models share the same structure: number of attributes, class count,
     * attribute cardinalities, order, and parent relationships.
     * </p>
     *
     * @param models An array of local {@link WDPT} models to be fused.
     * @return A new global {@link WDPT} model resulting from the fusion of all local models.
     */
    public Model fusion(Model[] models) {
        // Use the first model in the array as the starting point for fusion.
        // This assumes that all models have the same structure.
        WDPT reference = (WDPT) models[0];

        List<wdBayesParametersTree> fusedTrees = new ArrayList<>();
        List<AbstractClassifier> classifiers = reference.getClassifiers();
        List<Minimizer> minimizers = reference.getMinimizers();
        List<int[]> combinations = reference.getCombinations();
        List<Map<String, Integer>> classMaps = reference.getSyntheticClassMaps();
        List<ObjectiveFunction> functions = reference.getFunctions();

        int nTrees = reference.getTrees().size();

        for (int t = 0; t < nTrees; t++) {
            wdBayesParametersTree fused = reference.getTrees().get(t);

            if (fuseParameters) {
                this.fuseParameters(fused, models, t);
            }

            if (fuseProbabilities) {
                this.fuseProbabilities(fused, models, t);
            }

            fusedTrees.add(fused);
        }

        return new WDPT(fusedTrees, classifiers, minimizers, combinations, classMaps, functions);
    }

    /**
     * Fuses the parameters from all input models into the global model
     * by computing the element-wise average of the flattened parameter vectors.
     * <p>
     * Each {@link WDPT} model contains a flattened parameter array via {@link wdBayesParametersTree},
     * which represents all the logarithm probability weights used by the classifier. This method
     * aggregates them across all clients and sets the result into the global model.
     *
     * @param globalTree The global {@link wdBayesParametersTree} to store the fused parameters.
     * @param models The array of {@link WDPT} local models to be fused.
     * @param index Index of the tree to be fused (one per combination).
     */
    private void fuseParameters(wdBayesParametersTree globalTree, Model[] models, int index) {
        // Initialize an empty vector to accumulate parameter sums
        int length = globalTree.getParameters().length;
        double[] globalParameters = new double[length];

        // Sum all parameter vectors from local models element-wise
        for (Model model : models) {
            WDPT localModel = (WDPT) model;
            wdBayesParametersTree localTree = localModel.getTrees().get(index);
            double[] localParameters = localTree.getParameters();

            // Add local parameters to the global accumulator
            globalParameters = MathArrays.ebeAdd(globalParameters, localParameters);
        }

        // Compute the average by scaling the sum
        double val = 1.0 / models.length;
        globalParameters = MathArrays.scale(val, globalParameters);

        // Inject the averaged parameters into the global model
        globalTree.copyParameters(globalParameters);
    }

    /**
     * Fuses the conditional and class probabilities from all input models
     * into the global model by averaging them node-wise and normalizing.
     * <p>
     * This method fuses both the attribute-level conditional distributions stored
     * in the {@code xyCount} arrays of each {@link wdBayesNode}, and the class prior
     * probabilities stored in {@code classCounts}. It assumes all models share
     * the same structure and that the probabilities are in linear scale.
     * </p>
     *
     * @param globalTree The global {@link wdBayesParametersTree} that will store the fused probabilities.
     * @param models The array of {@link WDPT} local models to be fused.
     * @param index Index of the tree to be fused (one per combination).
     */
    private void fuseProbabilities(wdBayesParametersTree  globalTree, Model[] models, int index) {
        // Access number of attributes and classes
        int numAttributes = globalTree.getNAttributes();
        int nc = globalTree.getNc();

        // Fuse each attribute's conditional distributions
        for (int u = 0; u < numAttributes; u++) {
            wdBayesNode globalNode = globalTree.wdBayesNode_[u];
            wdBayesNode[] localNodes = new wdBayesNode[models.length];

            // Collect corresponding nodes from all local models
            for (int i = 0; i < models.length; i++) {
                WDPT localModel = (WDPT) models[i];
                wdBayesParametersTree localTree = localModel.getTrees().get(index);
                localNodes[i] = localTree.wdBayesNode_[u];
            }

            int paramsPerAttVal = globalNode.paramsPerAttVal;
            fuseNodeProbabilities(globalNode, localNodes, nc, paramsPerAttVal);
        }

        // Fuse and normalize the class prior probabilities
        fuseClassProbabilities(globalTree, models, index);
    }

    /**
     * Fuses the class probabilities from all input models
     * into the global model by averaging and normalizing them.
     * <p>
     * The {@code classCounts} field in each {@link wdBayesParametersTree} is assumed
     * to store normalized class probabilities (i.e., they sum to 1). This method computes
     * the element-wise average of these class probabilities across all models and then 
     * re-normalizes the result to ensure the final vector still sums to 1.
     * </p>
     * <p>
     * This step is essential when fusing class priors in federated Bayesian networks,
     * ensuring a coherent and valid global prior distribution.
     * </p>
     *
     * @param globalTree The global {@link wdBayesParametersTree} to store the fused class probabilities.
     * @param models The array of {@link WDPT} local models to be fused.
     * @param index Index of the tree to be fused (one per combination).
     */
    private void fuseClassProbabilities(wdBayesParametersTree globalTree, Model[] models, int index) {
        // Access the global model's classCounts array length
        int length = globalTree.getClassCounts().length;
        double[] globalClassProbs = new double[length];

        // Sum the class probability vectors from all models
        for (Model model : models) {
            WDPT localModel = (WDPT) model;
            wdBayesParametersTree localTree = localModel.getTrees().get(index);
            double[] localClassProbs = localTree.getClassCounts(); // Assumed to be probabilities
            globalClassProbs = MathArrays.ebeAdd(globalClassProbs, localClassProbs);
        }

        // Compute the average of the summed probabilities
        double val = 1.0 / models.length;
        globalClassProbs = MathArrays.scale(val, globalClassProbs);

        // Normalize to ensure the probabilities sum to one and store them in the global model
        globalTree.classCounts = MathArrays.normalizeArray(globalClassProbs, 1);
    }

    /**
     * Recursively fuses the conditional probabilities stored in {@code xyCount} arrays
     * from multiple local {@link wdBayesNode} instances into a global node.
     * <p>
     * It averages the conditional probabilities (assumed to be in logarithm scale),
     * and normalizes them per class to ensure that for each class {@code y}, the conditional
     * distribution remains a valid probability distribution (sums to 1).
     * </p>
     * <p>
     * This method operates recursively over the trie structure of each node, assuming that the
     * structure (children, scheme, parameters' shape) has already been aligned across all models.
     * </p>
     *
     * @param globalNode The {@link wdBayesNode} into which the fused probabilities will be stored.
     * @param localNodes An array of local {@link wdBayesNode}s to be fused.
     * @param nc The number of class values.
     * @param paramsPerAttVal The number of values for the attribute represented by this node.
     */
    private void fuseNodeProbabilities(wdBayesNode globalNode, wdBayesNode[] localNodes, int nc, int paramsPerAttVal) {
        // Safety check: skip if node is null or has no data to fuse
        if (globalNode == null || localNodes[0] == null || globalNode.xyCount == null)  {
            return;
        }

        int blockSize = nc * paramsPerAttVal;
        double[] globalProbabilities = new double[blockSize];

        // Step 1: Convert logarithm probabilities to linear space and average
        for (int i = 0; i < blockSize; i++) {
            for (wdBayesNode localNode : localNodes) {
                // Accumulate in linear space
                double probability = localNode.xyCount[i];
                globalProbabilities[i] += Math.exp(probability);
            }

            // Average
            globalProbabilities[i] /= localNodes.length;
        }

        // Step 2: Normalize each class block
        for (int y = 0; y < nc; y++) {
            double sum = 0.0;

            // Sum conditional probability across attribute values
            for (int x = 0; x < paramsPerAttVal; x++) {
                int idx = y * paramsPerAttVal + x;
                sum += globalProbabilities[idx]; // Row-major layout
            }

            // Normalize each entry for class label
            if (sum > 0) {
                for (int x = 0; x < paramsPerAttVal; x++) {
                    int idx = y * paramsPerAttVal + x;
                    double probability = Math.max(globalProbabilities[idx] / sum, MIN_PROBABILITY);
                    globalProbabilities[idx] = Math.log(probability);
                }
            }
        }

        // Step 3: Store back into global node
        System.arraycopy(globalProbabilities, 0, globalNode.xyCount, 0, blockSize);

        // Step 4: Recurse into children
        if (globalNode.children != null) {
            for (int i = 0; i < globalNode.children.length; i++) {
                if (globalNode.children[i] == null) {
                    // Skip if the global child node doesn't exist
                    continue;
                }

                // Collect the corresponding children from all local nodes
                wdBayesNode[] nextLocals = new wdBayesNode[localNodes.length];

                for (int j = 0; j < localNodes.length; j++) {
                    // Get the i-th child of each local node, if it exists
                    nextLocals[j] = (localNodes[j] != null && localNodes[j].children != null) ? localNodes[j].children[i] : null;
                }

                // Recurse
                fuseNodeProbabilities(
                    globalNode.children[i],
                    nextLocals,
                    nc,
                    globalNode.children[i].paramsPerAttVal
                );
            }
        }
    }
}
