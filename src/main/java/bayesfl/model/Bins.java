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
 * Package containing models related with federated Bayesian networks.
*/
package bayesfl.model;

/**
 * Local application imports.
 */
import bayesfl.data.Data;

/**
 * A class representing a discrezation model.
 */
public class Bins implements Model {

    /**
     * The cut points.
     */
    private double[][] cutPoints;

    /**
     * Constructor.
     * 
     * @param points The cut points.
     */
    public Bins(double[][] cutPoints) {
        this.cutPoints = cutPoints;
    }

    /**
     * Gets the model. In this case, the model is the cut points.
     * 
     * @return The model.
     */
    public double[][] getModel() {
        return this.cutPoints;
    }

    /**
     * Sets the model. This method is unused and throws an exception if called.
     * 
     * @param model The model.
     * @throws UnsupportedOperationException If the method is called.
     */
    public void setModel(Object model) {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Saves the statistics of the model. This method is unused and throws an exception if called.
     * 
     * @param operation The operation.
     * @param epoch The epoch.
     * @param path The path.
     * @param nClients The number of clients.
     * @param id The identifier.
     * @param data The data.
     * @param iteration The iteration.
     * @param time The time.
     * @throws UnsupportedOperationException If the method is called.
     */
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Gets the score of the model. This method is unused and throws an exception if called.
     *
     * @return The score of the model.
     */
    public double getScore() {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Computes the score of the model. This method is unused and throws an exception if called.
     * 
     * @param data The data.
     * @return The score of the model.
     */
    public double getScore(Data data) {
        throw new UnsupportedOperationException("Method not implemented");
    }
}
