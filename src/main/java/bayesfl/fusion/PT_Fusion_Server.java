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
        int length = model.getModel().length;
        double[] global = new double[length];

        for (int i = 0; i < models.length; i++) {
            // Add the local parameters to the global parameters
            model = (PT) models[i];
            double local[] = model.getModel();
            global = MathArrays.ebeAdd(global, local);
        }

        // Compute the mean of the parameters
        double val = 1.0 / models.length;
        global = MathArrays.scale(val, global);

        PT fused = new PT(global, null);
        return fused;
    }
}
