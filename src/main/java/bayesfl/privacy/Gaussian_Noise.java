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
 *    Gaussian_Noise.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.privacy;

import java.util.Random;

/**
 * Adds Gaussian noise for (ε, δ)-differential privacy.
 * <p>
 * This class implements the Gaussian mechanism, which provides (ε, δ)-DP guarantees.
 * The noise is drawn from a Gaussian(0, σ²) distribution, where σ is computed from the L2-sensitivity,
 * ε and δ.
 * </p>
 */
public class Gaussian_Noise implements NoiseGenerator {

    private double sigma;
    private double epsilon;
    private double delta;
    private double sensitivity;

    /**
     * Constructs a Gaussian noise generator.
     *
     * @param epsilon     the privacy budget ε (must be > 0)
     * @param delta       the failure probability δ (must be in (0, 1))
     * @param sensitivity the L2 sensitivity Δ of the function
     */
    public Gaussian_Noise(double epsilon, double delta, double sensitivity) {
        if (epsilon <= 0) throw new IllegalArgumentException("Epsilon must be positive.");
        if (delta <= 0 || delta >= 1) throw new IllegalArgumentException("Delta must be in (0, 1).");

        this.epsilon = epsilon;
        this.delta = delta;
        this.sensitivity = sensitivity;
        this.sigma = computeGaussianScale(sensitivity, epsilon, delta);
    }

    /**
     * Sets the sensitivity of the noise generator.
     *
     * @param newSensitivity the new sensitivity value
     */
    @Override
    public void setSensitivity(double newSensitivity) {
        this.sensitivity = newSensitivity;
        this.sigma = computeGaussianScale(newSensitivity, this.epsilon, this.delta);
    }

    /**
     * Adds Gaussian noise to a scalar value.
     *
     * @param value the original value
     * @return the privatized value with Gaussian noise added
     */
    @Override
    public double privatize(double value) {
        return value + sigma * rng.nextGaussian();
    }

    /**
     * Adds Gaussian noise to each element of a vector.
     *
     * @param values the original array of values
     * @return a new array where each element has Gaussian noise added
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
     * Computes the Gaussian scale parameter σ for (ε, δ)-DP.
     * <p>
     * Uses the formula: σ = (Δ * sqrt(2 * ln(1.25 / δ))) / ε
     * </p>
     *
     * @param sensitivity the L2 sensitivity Δ
     * @param epsilon     the privacy budget ε
     * @param delta       the failure probability δ
     * @return the Gaussian scale parameter σ
     */
    private static double computeGaussianScale(double sensitivity, double epsilon, double delta) {
        return sensitivity * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
    }
}
