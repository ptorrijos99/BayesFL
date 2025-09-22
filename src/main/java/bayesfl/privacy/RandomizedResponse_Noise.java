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
 *    RandomizedResponse_Noise.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.privacy;

/**
 * Implements the Randomized Response mechanism for ε-differential privacy on binary attributes.
 * <p>
 * This mechanism is used to perturb binary responses (true/false) to protect individual privacy.
 * A "true" response is reported truthfully with high probability, and flipped to "false" with low probability.
 * The probabilities are controlled by the privacy budget ε.
 * </p>
 */
public class RandomizedResponse_Noise implements NoiseGenerator {

    private final double epsilon;
    private final double p_truth; // Probability of telling the truth
    private final double q_lie;   // Probabilidad of lying

    /**
     * Constructs a Randomized Response noise generator.
     *
     * @param epsilon the privacy budget (ε), must be > 0
     */
    public RandomizedResponse_Noise(double epsilon) {
        if (epsilon <= 0) throw new IllegalArgumentException("Epsilon must be positive.");
        this.epsilon = epsilon;
        this.p_truth = Math.exp(epsilon) / (Math.exp(epsilon) + 1);
        this.q_lie = 1 / (Math.exp(epsilon) + 1);
    }

    /**
     * Applies Randomized Response to a boolean value.
     *
     * @param originalValue the original boolean value (e.g., true if an edge exists)
     * @return the privatized boolean value
     */
    public boolean flip(boolean originalValue) {
        if (rng.nextDouble() < p_truth) {
            return originalValue; // Dice la verdad
        } else {
            return !originalValue; // Miente
        }
    }

}
