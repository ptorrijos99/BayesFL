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
 *    Laplace_Noise.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.privacy;

import java.util.Random;

/**
 * Adds Laplace noise for ε-differential privacy.
 * <p>
 * This class implements the Laplace mechanism for injecting noise into scalar or vector values,
 * providing pure ε-DP guarantees. The noise is drawn from a Laplace(0, b) distribution, where
 * the scale parameter {@code b = sensitivity / ε}.
 * </p>
 */
public class Laplace_Noise implements NoiseGenerator {

    private double scale;
    private double epsilon;
    private double sensitivity;

    /**
     * Constructs a Laplace noise generator.
     *
     * @param epsilon     the privacy budget (ε), must be > 0
     * @param sensitivity the L1 sensitivity (Δ) of the function to which noise will be added
     */
    public Laplace_Noise(double epsilon, double sensitivity) {
        if (epsilon <= 0) throw new IllegalArgumentException("Epsilon must be positive.");
        this.epsilon = epsilon;
        this.sensitivity = sensitivity;
        this.scale = sensitivity / epsilon;
    }

    /**
     * Sets the sensitivity of the noise generator.
     *
     * @param newSensitivity the new sensitivity value
     */
    @Override
    public void setSensitivity(double newSensitivity) {
        this.sensitivity = newSensitivity;
        this.scale = newSensitivity / this.epsilon;
    }

    /**
     * Adds Laplace noise to a scalar value.
     *
     * @param value the original value
     * @return the privatized value with Laplace noise added
     */
    @Override
    public double privatize(double value) {
        return value + laplaceNoise(scale, rng);
    }

    /**
     * Adds Laplace noise to each element of a vector.
     *
     * @param values the original array of values
     * @return a new array where each element has Laplace noise added
     */
    @Override
    public double[] privatize(double[] values) {
        double[] noisy = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            noisy[i] = privatize(values[i]);
        }
        return noisy;
    }

    /**
     * Draws a sample from a Laplace(0, scale) distribution.
     *
     * @param scale the scale parameter b = Δ / ε
     * @param rng   the random number generator
     * @return a Laplace-distributed noise sample
     */
    public static double laplaceNoise(double scale, Random rng) {
        double u = rng.nextDouble() - 0.5;
        return -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
    }
}

