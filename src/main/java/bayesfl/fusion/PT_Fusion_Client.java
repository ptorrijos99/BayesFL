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
import EBNC.wdBayes;
import weka.classifiers.meta.FilteredClassifier;

/**
 * Local application imports.
 */
import bayesfl.model.Model;
import bayesfl.model.PT;

/**
 * A class representing a fusion method for class-conditional Bayesian networks in the client.
 */
public class PT_Fusion_Client implements Fusion {

    /**
     * Perform the fusion of two models.
     * 
     * @param model1 The first model to fuse.
     * @param model2 The second model to fuse.
     * @return The global model fused.
     */
    public Model fusion(Model model1, Model model2) {
        // Get the classifier from the client and the parameters from the server
        PT client = (PT) model1;
        PT server = (PT) model2;
        FilteredClassifier classifier = client.getClassifier();
        double[] parameters = server.getModel();

        // Copy the parameters from the server to the client before building the model
        wdBayes algorithm = (wdBayes) classifier.getClassifier();
        algorithm.dParameters_.copyParameters(parameters);

        return new PT(parameters, classifier);
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
}
