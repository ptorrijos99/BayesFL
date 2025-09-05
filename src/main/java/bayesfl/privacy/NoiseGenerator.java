/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2025 Universidad de Castilla-La Mancha, España
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
 *    NoiseGenerator.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.privacy;

import java.util.Random;

/**
 * Interface for injecting Differential Privacy (DP) noise into numerical values.
 * <p>
 * Implementations of this interface define how to add noise to individual values or arrays of values
 * according to a specific DP mechanism (e.g., Laplace, Gaussian). This abstraction allows models
 * to apply DP without depending on the underlying privacy mechanism.
 * </p>
 * <p>
 * Typical use cases include privatizing counts, probabilities, or parameters in federated or
 * distributed learning settings.
 * </p>
 */
public interface NoiseGenerator {

    /**
     * The random number generator used for generating noise.
     * <p>
     * It can be replaced with a more secure or reproducible generator if needed.
     * </p>
     */
    Random rng = new Random();

    /**
     * Adds differential privacy noise to a single scalar value.
     *
     * @param value the original (non-private) value
     * @return the privatized value with noise added
     */
    double privatize(double value);

    /**
     * Adds differential privacy noise independently to each element of a vector.
     *
     * @param values the original (non-private) array of values
     * @return a new array where each element has DP noise added
     */
    double[] privatize(double[] values);

    /**
     * Sets the sensitivity of the function being privatized.
     * <p>
     * This is used to adjust the scale of the noise based on the sensitivity of the data.
     * </p>
     *
     * @param newSensitivity the new sensitivity value
     */
    void setSensitivity(double newSensitivity);
}

