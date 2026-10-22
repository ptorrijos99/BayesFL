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
 *    SecAggLaplace_Noise.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.privacy;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.RandomGeneratorFactory;

/**
 * Laplace mechanism split across the clients of a secure-aggregation round.
 * <p>
 * With secure aggregation the server only learns the sum of the clients'
 * contributions, so the sum is what must be ε-DP. Each of the {@code n}
 * clients adds {@code G1 - G2} with {@code G1, G2 ~ Gamma(1/n, b)}: by the
 * infinite divisibility of the Laplace distribution, the sum of the {@code n}
 * shares is exactly one {@code Laplace(0, b)} draw, {@code b = sensitivity / ε}.
 * The aggregate therefore carries the same noise, and the same pure ε-DP
 * guarantee, as a single central release, instead of {@code n} full draws.
 * </p>
 * <p>
 * The guarantee assumes every client contributes its share and none of them
 * reveals it to the server; a single share is not private on its own.
 * </p>
 */
public class SecAggLaplace_Noise implements NumericNoiseGenerator {

    private final double epsilon;
    private final int nShares;
    private double sensitivity;
    private GammaDistribution gamma;

    /**
     * Constructs a noise generator that returns one client's share.
     *
     * @param epsilon     the privacy budget (ε) of the aggregate, must be > 0
     * @param sensitivity the L1 sensitivity (Δ) of the aggregated function
     * @param nShares     the number of clients whose shares are summed, must be >= 1
     */
    public SecAggLaplace_Noise(double epsilon, double sensitivity, int nShares) {
        if (epsilon <= 0) throw new IllegalArgumentException("Epsilon must be positive.");
        if (nShares < 1) throw new IllegalArgumentException("The number of shares must be at least 1.");
        this.epsilon = epsilon;
        this.nShares = nShares;
        setSensitivity(sensitivity);
    }

    @Override
    public void setSensitivity(double newSensitivity) {
        this.sensitivity = newSensitivity;
        this.gamma = new GammaDistribution(RandomGeneratorFactory.createRandomGenerator(rng),
                1.0 / nShares, newSensitivity / epsilon);
    }

    @Override
    public double privatize(double value) {
        return value + gamma.sample() - gamma.sample();
    }

    @Override
    public double[] privatize(double[] values) {
        double[] noisy = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            noisy[i] = privatize(values[i]);
        }
        return noisy;
    }

    @Override
    public boolean isShare() {
        return true;
    }

    /**
     * @return the number of clients whose shares add up to the aggregate noise
     */
    public int getNShares() {
        return nShares;
    }
}
