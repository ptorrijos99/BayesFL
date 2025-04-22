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
 * Third party imports.
 */
import weka.filters.unsupervised.attribute.Discretize;

/**
 * A dummy class to set the cut points of the discretization filter.
 */
public class Dummy extends Discretize {

    /**
     * Auxiliar variable to store the cut points.
     */
    private double[][] cutPoints;

    /**
     * Generate the cutpoints for each attribute.
     * In this case, it does nothing because the cut points are set manually.
     */
    protected void calculateCutPoints() {
        this.m_CutPoints = this.cutPoints;
    }

    /**
     * Set the cut points.
     *
     * @param cutPoints The cut points to set.
     */
    public void setCutPoints(double[][] cutPoints) {
        // We cannot set the cut points directly because they
        // are somewhere reset before building the classifier
        this.cutPoints = cutPoints;
    }
}
