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
import org.apache.commons.math3.util.MathArrays;

/**
 * Local application imports.
 */
import bayesfl.model.Model;
import bayesfl.model.PT;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.estimators.DiscreteEstimator;

class DummyNB extends NaiveBayes {
    public void setDistributions(DiscreteEstimator[][] m_Distributions) {
        this.m_Distributions = m_Distributions;
    }

    public void setClassDistribution(DiscreteEstimator m_ClassDistribution) {
        this.m_ClassDistribution = m_ClassDistribution;
    }
}

/**
 * A class representing a fusion method for class-conditional Bayesian networks in the server.
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
    public Model fusion(Model[] models) {
        // Initialize the global model in a new array
        // to avoid modifying the original first model
        PT model = (PT) models[0];

        // WEKA NaiveBayes
        if (model.getModel() == null) {
            int numAtts = model.getM_Distributions().length;
            int numClasses = model.getM_Distributions()[0].length;

            // Initialize the distributions
            DiscreteEstimator m_ClassDistribution = new DiscreteEstimator(numClasses, true);
            DiscreteEstimator[][] m_Distributions = new DiscreteEstimator[numAtts][numClasses];

            for (int i = 0; i < numAtts; i++) {
                for (int j = 0; j < numClasses; j++) {
                    int numSymbols = ((DiscreteEstimator) model.getM_Distributions()[i][j]).getNumSymbols();
                    m_Distributions[i][j] = new DiscreteEstimator(numSymbols, true);
                }
            }

            System.out.println("Fusing " + models.length + " models");
            System.out.println("Antes: " + m_ClassDistribution);

            for (Model value : models) {
                model = (PT) value;

                try {
                    // Add the class distribution of all the clients
                    DiscreteEstimator classDist = (DiscreteEstimator) model.getM_ClassDistribution();
                    //classDist.
                    m_ClassDistribution.aggregate(classDist);

                    System.out.println("  Suma: " + model.getM_ClassDistribution());

                    // Add each var distribution of all the clients
                    for (int j = 0; j < numAtts; j++) {
                        for (int k = 0; k < numClasses; k++) {
                            m_Distributions[j][k].aggregate((DiscreteEstimator) model.getM_Distributions()[j][k]);
                        }
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            System.out.println("Después: " + m_ClassDistribution);

            DummyNB naiveBayes = new DummyNB();
            naiveBayes.setDistributions(m_Distributions);
            naiveBayes.setClassDistribution(m_ClassDistribution);

            FilteredClassifier naiveBayesFC = new FilteredClassifier();
            naiveBayesFC.setClassifier(naiveBayes);
            naiveBayesFC.setFilter(model.getClassifier().getFilter());

            return new PT(m_ClassDistribution, m_Distributions, naiveBayesFC);
        }
        // wdBayes (NBw, NBb and NBe algorithms)
        else {
            int length = model.getModel().length;
            double[] global = new double[length];

            for (Model value : models) {
                // Add the local parameters to the global parameters
                model = (PT) value;
                double[] local = model.getModel();
                global = MathArrays.ebeAdd(global, local);
            }

            // Compute the mean of the parameters
            double val = 1.0 / models.length;
            global = MathArrays.scale(val, global);

            return new PT(global, null);
        }
    }
}
