/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2026 Universidad de Castilla-La Mancha, España
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
 * Package containing algorithms related with differential privacy for
 * federated Bayesian networks.
 */
package bayesfl.algorithms.dp;

/**
 * Third party imports.
 */
import bayesfl.privacy.ZCdpGaussianCalibration;

/**
 * Immutable DP-SGD configuration for the discriminative channel.
 *
 * <p>The noise multiplier {@link #sigma()} is derived once at construction
 * time via {@link ZCdpGaussianCalibration#noiseMultiplier(double, double, long)}
 * over the total number of full-batch steps ({@code rounds * localSteps}), so
 * that the privacy accounting matches the actual number of DP-SGD steps that
 * will be taken across the whole run. When {@code epsilonParam} is infinite
 * (no privacy budget requested), {@code sigma} is fixed to {@code 0.0} and no
 * noise is added by {@link DPSGDOptimizer}.</p>
 */
public final class DPSGDConfig {

    /** Target per-tree privacy budget epsilon; {@code Double.POSITIVE_INFINITY} disables DP noise. */
    private final double epsilonParam;

    /** Target privacy failure probability delta. */
    private final double delta;

    /** Per-sample L2 clipping norm C for the concatenated cross-tree gradient. */
    private final double clipC;

    /** Fixed learning rate for the DP-SGD parameter step. */
    private final double learningRate;

    /** Number of federated rounds this configuration will be used across. */
    private final int rounds;

    /** Number of full-batch DP-SGD steps performed per round. */
    private final int localSteps;

    /** Seed for the shared noise RNG. */
    private final long seed;

    /** Precomputed Gaussian noise multiplier; {@code 0.0} when DP is disabled. */
    private final double sigma;

    /**
     * Builds an immutable DP-SGD configuration and calibrates {@link #sigma()}
     * for the total number of steps ({@code rounds * localSteps}).
     *
     * @param epsilonParam target per-tree epsilon; {@code Double.POSITIVE_INFINITY} disables noise.
     * @param delta        target delta.
     * @param clipC        per-sample L2 clipping norm.
     * @param rounds       number of federated rounds.
     * @param localSteps   number of full-batch DP-SGD steps per round.
     * @param learningRate fixed learning rate.
     * @param seed         seed for the shared noise RNG.
     */
    public DPSGDConfig(double epsilonParam, double delta, double clipC,
                        int rounds, int localSteps, double learningRate, long seed) {
        this.epsilonParam = epsilonParam;
        this.delta = delta;
        this.clipC = clipC;
        this.rounds = rounds;
        this.localSteps = localSteps;
        this.learningRate = learningRate;
        this.seed = seed;
        long T = (long) rounds * localSteps;
        this.sigma = Double.isInfinite(epsilonParam)
                ? 0.0
                : ZCdpGaussianCalibration.noiseMultiplier(epsilonParam, delta, T);
    }

    /** @return the configured per-tree epsilon (may be infinite). */
    public double epsilonParam() { return epsilonParam; }

    /** @return the configured delta. */
    public double delta()        { return delta; }

    /** @return the per-sample L2 clipping norm C. */
    public double clipC()        { return clipC; }

    /** @return the number of federated rounds. */
    public int rounds()          { return rounds; }

    /** @return the number of full-batch DP-SGD steps per round. */
    public int localSteps()      { return localSteps; }

    /** @return the fixed learning rate. */
    public double learningRate() { return learningRate; }

    /** @return the shared noise RNG seed. */
    public long seed()           { return seed; }

    /** @return the calibrated Gaussian noise multiplier ({@code 0.0} when DP is disabled). */
    public double sigma()        { return sigma; }

    /** @return {@code true} when a finite epsilon was requested (DP noise is added). */
    public boolean enabled()     { return !Double.isInfinite(epsilonParam); }
}
