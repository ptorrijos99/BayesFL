/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
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
 *    ScoreImprovement.java
 *    Copyright (C) 2025 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.convergence;

import bayesfl.model.Model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This convergence criterion checks if the mean score of the local models has improved.
 */
public class ScoreImprovement implements Convergence {

    private double lastScoreMean = Double.NEGATIVE_INFINITY;

    /**
     * Checks if the local models have converged by checking if the mean score of the local models has improved.
     *
     * @param localModels The local models of the clients.
     * @return True if the local models have converged, false otherwise.
     */
    @Override
    public boolean checkConvergence(Model[] localModels) {
        boolean convergence = true;

        double scoreMean = 0;
        for (Model m : localModels) {
            scoreMean += m.getScore();
        }
        scoreMean /= localModels.length;

        if (scoreMean > lastScoreMean) {
            convergence = false;
        }

        System.out.println("\n\n Score mean: " + scoreMean + " Last score mean: " + lastScoreMean + " Convergence: " + convergence);

        lastScoreMean = scoreMean;

        return convergence;
    }
}
