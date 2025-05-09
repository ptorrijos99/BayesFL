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

/**
 * Third-party imports.
 */
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import objectiveFunction.ObjectiveFunction;
import optimize.Minimizer;
import weka.classifiers.AbstractClassifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Local application imports.
 */
import bayesfl.model.Model;
import bayesfl.model.WDPT;

/**
 * A class representing a fusion method for class-conditional Bayesian networks in the client.
 */
public class WDPT_Fusion_Client implements Fusion {

    /**
     * Whether to fuse the parameters or not.
     */
    private final boolean fuseParameters;

    /**
     * Whether to fuse the probabilities or not.
     */
    private final boolean fuseProbabilities;

    /** Constructor.
     *
     * @param fuseParameters Whether to fuse the parameters or not.
     * @param fuseProbabilities Whether to fuse the probabilities or not.
     */
    public WDPT_Fusion_Client(boolean fuseParameters, boolean fuseProbabilities) {
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
        WDPT local = (WDPT) model1;
        WDPT global = (WDPT) model2;

        List<wdBayesParametersTree> fusedTrees = new ArrayList<>();
        List<AbstractClassifier> classifiers = local.getClassifiers();  // Reuse local classifiers
        List<Minimizer> minimizers = local.getMinimizers();
        List<int[]> combinations = local.getCombinations();
        List<Map<String, Integer>> classMaps = local.getSyntheticClassMaps();
        List<ObjectiveFunction> functions = local.getFunctions();

        List<wdBayesParametersTree> localTrees = local.getTrees();
        List<wdBayesParametersTree> globalTrees = global.getTrees();

        for (int i = 0; i < localTrees.size(); i++) {
            wdBayesParametersTree localTree = localTrees.get(i);
            wdBayesParametersTree globalTree = globalTrees.get(i);

            // Fuse parameters if enabled
            if (fuseParameters) {
                localTree.copyParameters(globalTree.getParameters());
            }

            // Fuse log-probabilities (structure and node-wise) if enabled
            if (fuseProbabilities) {
                copyLogProbsFromTree(globalTree, localTree);
            }

            fusedTrees.add(localTree);  // trees are now updated in place
        }

        return new WDPT(fusedTrees, classifiers, minimizers, combinations, classMaps, functions);
    }

    /**
     * Fusion of an array of models. This method is unused and throws an exception if called.
     * 
     * @param models The array of models to fuse.
     * @return The global model fused.
     */
    public Model fusion(Model[] models) {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Copies the logarithm probabilities from a source tree-based parameter storage
     * to a target tree-based parameter storage. This includes class probabilities and
     * the trie-based conditional probabilities for all attributes.
     *
     * Note: This method assumes that both trees have identical structure, 
     * including attribute order, parents, and children layout.
     *
     * @param source The source parameters tree containing logarithm probabilities.
     * @param target The target parameters tree to receive the copied logarithm probabilities.
     */
    private static void copyLogProbsFromTree(wdBayesParametersTree source, wdBayesParametersTree target) {
        // Copy class probabilities
        System.arraycopy(source.classCounts, 0, target.classCounts, 0, source.classCounts.length);

        // Traverse and copy each attribute's trie
        int numAttributes = source.getNAttributes();

        for (int u = 0; u < numAttributes; u++) {
            wdBayesNode sourceNode = source.wdBayesNode_[u];
            wdBayesNode targetNode = target.wdBayesNode_[u];
            copyLogProbsRecursive(sourceNode, targetNode);
        }
    }

    /**
     * Recursively copies the logarithm probabilities array
     * from a source node to a target node.
     *
     * This method assumes that the node structures are aligned
     * and initialized in both source and target trees.
     *
     * @param source The source node whose logarithm probabilities will be copied.
     * @param target The target node that will receive the copied values.
     */
    private static void copyLogProbsRecursive(wdBayesNode source, wdBayesNode target) {
        if (source == null || target == null) {
            return;
        }
    
        // Copy the logarithm probabilities from the source node to the target node
        if (source.xyCount != null && target.xyCount != null) {
            // Note: Although the field is named "xyCount", it actually stores the logarithm probabilities
            System.arraycopy(source.xyCount, 0, target.xyCount, 0, source.xyCount.length);
        }
    
        // Recursively copy children
        if (source.children != null && target.children != null) {
            for (int i = 0; i < source.children.length; i++) {
                if (source.children[i] != null && target.children[i] != null) {
                    copyLogProbsRecursive(source.children[i], target.children[i]);
                }
            }
        }
    }
}
