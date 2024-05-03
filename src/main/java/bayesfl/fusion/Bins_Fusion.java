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
import java.util.Arrays;

/**
 * Third-party imports.
 */
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.MathArrays;

/**
 * Local application imports.
 */
import bayesfl.model.Bins;
import bayesfl.model.Model;

/**
 * A class to perform the fusion of bins, that is, cut points.
 */
public class Bins_Fusion implements Fusion {

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
        // Use the first model to get the number of attributes
        Bins model = (Bins) models[0];
        int numClients = models.length;
        int numAttributes = model.getModel().length;

        // First step, compute the mode number of bins for each attribute

        double[][] numBins = new double[numAttributes][numClients];

        for (int i = 0; i < numClients; i++) {
            // Get the cut points of the model
            model = (Bins) models[i];
            double[][] cutPoints = model.getModel();

            for (int j = 0; j < numAttributes; j++) {
                // Skip null values because it means that no
                // cut points were located for the attribute
                if (cutPoints[j] != null) {
                    numBins[j][i] = cutPoints[j].length;
                }
            }
        }

        // Compute the modes of the number of bins
        int[] modes = this.mode(numBins);

        // Second step, compute the mean of the cut points of each 
        // attribute, but use only clients with the same bin count

        double[][] global = new double[numAttributes][];

        for (int i = 0; i < numAttributes; i++) {
            // Get the number of bins for the attribute
            int size = modes[i];

            // Skip attributes with no cut points
            if (size == 0) {
                continue;
            }

            // Initialize the array to store the global cut points
            global[i] = new double[size];

            // Counter for the number of clients with the same number of bins
            double count = 0;

            for (int j = 0; j < numClients; j++) {
                // Get the cut points of the model
                model = (Bins) models[j];
                double[][] cutPoints = model.getModel();

                // Skip clients with no cut points for the attribute
                if (cutPoints[i] == null) {
                    continue;
                }

                if (cutPoints[i].length == size) {
                    // Add the cut points to the global model
                    double[] a = global[i];
                    double[] b = cutPoints[i];
                    global[i] = MathArrays.ebeAdd(a, b);
                    count++;
                }
            }

            // Compute the mean of the added cut points, which is
            // similar to scaling the sum by the count divided by
            // the number of clients with the same number of bins
            double val = 1 / count;
            double[] arr = global[i];
            global[i] = MathArrays.scale(val, arr);
        }

        return new Bins(global);
    }

    /**
     * Compute the modes of an array of arrays.
     * 
     * @param arrays The array of arrays.
     * @return The modes of the array of arrays.
     */
    private int[] mode(double[][] arrays) {
        int[] modes = new int[arrays.length];

        for (int i = 0; i < arrays.length; i++) {
            // Get the values of the array of arrays
            double values[] = arrays[i];

            // Compute the mode of the array and store the
            // first found because it is the most frequent
            double mode[] = StatUtils.mode(values);
            modes[i] = (int) mode[0];
        }

        return modes;
    }
}
