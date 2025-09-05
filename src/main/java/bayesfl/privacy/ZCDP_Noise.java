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
 *    ZCDP_Noise.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.privacy;

import java.util.Random;

/**
 * Adds Gaussian noise for zero-Concentrated Differential Privacy (zCDP).
 * <p>
 * This class implements the zCDP Gaussian mechanism, which provides tighter composition
 * bounds than standard (ε, δ)-DP. The noise is drawn from a Gaussian(0, σ²) distribution,
 * where σ is computed as {@code Δ / sqrt(2ρ)}.
 * </p>
 */
public class ZCDP_Noise implements NoiseGenerator {

    private double sigma;
    private double rho;
    private double sensitivity;

    /**
     * Constructs a zCDP noise generator.
     *
     * @param rho         the zCDP privacy parameter ρ (must be > 0)
     * @param sensitivity the L2 sensitivity Δ of the function
     */
    public ZCDP_Noise(double rho, double sensitivity) {
        if (rho <= 0) throw new IllegalArgumentException("zCDP parameter rho must be > 0.");

        this.rho = rho;
        this.sensitivity = sensitivity;
        this.sigma = computeZCDPScale(sensitivity, rho);
    }

    /**
     * Sets the sensitivity of the noise generator.
     *
     * @param newSensitivity the new sensitivity value
     */
    @Override
    public void setSensitivity(double newSensitivity) {
        this.sensitivity = newSensitivity;
        this.sigma = computeZCDPScale(newSensitivity, this.rho);
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
     * Computes the Gaussian scale parameter σ for zCDP.
     * <p>
     * Uses the formula: σ = Δ / sqrt(2ρ)
     * </p>
     *
     * @param sensitivity the L2 sensitivity Δ
     * @param rho         the zCDP parameter ρ
     * @return the Gaussian scale parameter σ
     */
    private static double computeZCDPScale(double sensitivity, double rho) {
        return sensitivity / Math.sqrt(2 * rho);
    }
}
