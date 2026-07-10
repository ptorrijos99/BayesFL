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

package bayesfl.privacy;

/**
 * Closed-form conversions between zero-concentrated DP (rho) and (epsilon, delta)-DP,
 * and the DP-SGD noise-multiplier calibration for full-batch Gaussian gradient steps.
 *
 * Per full-batch step, the Gaussian mechanism with L2 sensitivity C and noise std
 * (sigma * C) is (1 / (2 sigma^2))-zCDP; over T steps rho = T / (2 sigma^2).
 * The standard tight bound epsilon = rho + 2 sqrt(rho * ln(1/delta)) is inverted
 * in closed form.
 */
public final class ZCdpGaussianCalibration {

    private ZCdpGaussianCalibration() {}

    /** rho such that epsilon = rho + 2 sqrt(rho * ln(1/delta)). */
    public static double rhoForEpsilon(double epsilon, double delta) {
        if (epsilon <= 0) throw new IllegalArgumentException("epsilon must be positive");
        if (delta <= 0 || delta >= 1) throw new IllegalArgumentException("delta must be in (0,1)");
        double l = Math.log(1.0 / delta);
        double x = Math.sqrt(l + epsilon) - Math.sqrt(l); // x = sqrt(rho)
        return x * x;
    }

    /** epsilon = rho + 2 sqrt(rho * ln(1/delta)). */
    public static double epsilonForRho(double rho, double delta) {
        if (rho < 0) throw new IllegalArgumentException("rho must be non-negative");
        if (delta <= 0 || delta >= 1) throw new IllegalArgumentException("delta must be in (0,1)");
        double l = Math.log(1.0 / delta);
        return rho + 2.0 * Math.sqrt(rho * l);
    }

    /** Noise multiplier sigma to achieve (epsilon, delta) over totalSteps full-batch steps. */
    public static double noiseMultiplier(double epsilon, double delta, long totalSteps) {
        if (totalSteps <= 0) throw new IllegalArgumentException("totalSteps must be positive");
        double rho = rhoForEpsilon(epsilon, delta);
        return Math.sqrt(totalSteps / (2.0 * rho));
    }

    /** rho consumed by totalSteps full-batch steps at noise multiplier sigma. */
    public static double rhoSpent(long totalSteps, double sigma) {
        if (sigma <= 0) throw new IllegalArgumentException("sigma must be positive");
        return totalSteps / (2.0 * sigma * sigma);
    }
}
