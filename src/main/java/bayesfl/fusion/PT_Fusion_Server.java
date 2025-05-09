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
 * Standard imports.
 */
import java.util.*;

/**
 * Third-party imports.
 */
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.estimators.DiscreteEstimator;

/**
 * Local application imports.
 */
import bayesfl.model.Model;
import bayesfl.model.PT;

class DummyNB extends NaiveBayes {
    public void setDistributions(DiscreteEstimator[][] m_Distributions) {
        this.m_Distributions = m_Distributions;

        // Initialize this.m_Instances because it is used in the method distributionForInstance() to obtain weights
        ArrayList<Attribute> attInfo = new ArrayList<>();
        for (int i = 0; i < m_Distributions.length; i++) {
            attInfo.add(new Attribute("att" + i));
        }
        this.m_Instances = new Instances("Instances", attInfo, 0);
    }

    public void setClassDistribution(DiscreteEstimator m_ClassDistribution) {
        this.m_ClassDistribution = m_ClassDistribution;

        // Initialize this.m_NumClasses because it is used in the method distributionForInstance()
        this.m_NumClasses = m_ClassDistribution.getNumSymbols();
    }
}

/**
 * A class representing a fusion method for naive Bayes in the server.
 */
public class PT_Fusion_Server implements Fusion {

    /**
     * Perform the fusion of two models.
     * 
     * @param model1 The first model to fuse.
     * @param model2 The second model to fuse.
     * @return The global model fused.
     */
    public Model fusion(Model model1, Model model2) {
        Model[] models = {model1, model2};

        return fusion(models);
    }

    /**
     * Fusion several models.
     * 
     * @param models The array of models to fuse.
     * @return The global model fused.
     */
    @Override
    public Model fusion(Model[] models) {
        PT base = (PT) models[0];
        int numClassifiers = base.ensemble.size();
        List<AbstractClassifier> fusedEnsemble = new ArrayList<>();
        List<int[]> combinations = base.combinations;
        List<Map<String, Integer>> fusedClassMaps = new ArrayList<>();

        for (int i = 0; i < numClassifiers; i++) {
            // Collect all synthetic classes
            Set<String> allSyntheticClasses = new LinkedHashSet<>();
            for (Model m : models) {
                PT pt = (PT) m;
                allSyntheticClasses.addAll(pt.syntheticClassMaps.get(i).keySet());
            }

            List<String> classList = new ArrayList<>(allSyntheticClasses);
            Map<String, Integer> classMap = new LinkedHashMap<>();
            for (int idx = 0; idx < classList.size(); idx++) {
                classMap.put(classList.get(idx), idx);
            }

            int numClasses = classList.size();

            // Initialize fusion structures
            DiscreteEstimator fusedClassDist = new DiscreteEstimator(numClasses, true);
            List<DiscreteEstimator[]> fusedDistributions = new ArrayList<>();
            int numAtts = -1;

            for (Model m : models) {
                PT pt = (PT) m;
                FilteredClassifier filtered = (FilteredClassifier) pt.ensemble.get(i);
                NaiveBayes localNB = (NaiveBayes) filtered.getClassifier();
                Map<String, Integer> localMap = pt.syntheticClassMaps.get(i);

                // First time only: allocate distributions
                if (fusedDistributions.isEmpty()) {
                    numAtts = localNB.getConditionalEstimators().length;
                    for (int att = 0; att < numAtts; att++) {
                        DiscreteEstimator[] row = new DiscreteEstimator[numClasses];
                        for (int c = 0; c < numClasses; c++) {
                            DiscreteEstimator localEst = (DiscreteEstimator) localNB.getConditionalEstimators()[att][0];
                            row[c] = new DiscreteEstimator(localEst.getNumSymbols(), true);
                        }
                        fusedDistributions.add(row);
                    }
                }

                // Aggregate class distribution
                for (Map.Entry<String, Integer> entry : localMap.entrySet()) {
                    String label = entry.getKey();
                    int localIdx = entry.getValue();
                    int globalIdx = classMap.get(label);

                    double weight = localNB.getClassEstimator().getProbability(localIdx);
                    fusedClassDist.addValue(globalIdx, weight);

                    // Aggregate per-attribute conditional estimators
                    for (int att = 0; att < numAtts; att++) {
                        DiscreteEstimator localEstimator = (DiscreteEstimator) localNB.getConditionalEstimators()[att][localIdx];
                        try {
                            fusedDistributions.get(att)[globalIdx].aggregate(localEstimator);
                        } catch (Exception ignored) {}
                    }
                }
            }

            // Create fused classifier
            DummyNB fusedNB = new DummyNB();
            fusedNB.setDistributions(fusedDistributions.toArray(new DiscreteEstimator[0][]));
            fusedNB.setClassDistribution(fusedClassDist);
            fusedEnsemble.add(fusedNB);
            fusedClassMaps.add(classMap);
        }

        return new PT(fusedEnsemble, combinations, fusedClassMaps);
    }

}