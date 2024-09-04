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
 *    IterationEquality.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.convergence;

import bayesfl.model.Model;

import java.util.LinkedList;

/**
 * This convergence criterion checks if the local models have converged by checking if all the models of this iteration
 * are the same as the models of the previous iterations. The number of previous iterations to check is specified in
 * the constructor.
 */
public class IterationEquality implements Convergence {

    private final LinkedList<Model[]> lastLocalModels;

    private int maxModelsSaved;

    public IterationEquality(int maxModelsSaved) {
        this.maxModelsSaved = maxModelsSaved;

        lastLocalModels = new LinkedList<>();
    }

    /**
     * Checks if the local models have converged.
     *
     * @param localModels The local models of the clients.
     * @return True if the local models have converged, false otherwise.
     */
    @Override
    public boolean checkConvergence(Model[] localModels) {
        boolean convergence;
        if (maxModelsSaved < 0) maxModelsSaved = lastLocalModels.size();

        if (localModels[0].getScore() == 0) {
            convergence = checkModels(localModels);
        } else {
            convergence = checkStats(localModels);
        }

        lastLocalModels.add(localModels.clone());
        if (lastLocalModels.size() > maxModelsSaved)
            lastLocalModels.remove(0);

        return convergence;
    }

    private boolean checkStats(Model[] localModels) {
        if (lastLocalModels.isEmpty()) return false;

        iterations: for (int i = lastLocalModels.size()-1; (i >= 0) && (lastLocalModels.size()-1-i < maxModelsSaved); i--) {
            Model[] lastModel = lastLocalModels.get(i);

            for (int j = 0; j < localModels.length; j++) {
                // Checking if the score is the same
                if ((lastModel[j] == null) || !(Math.abs(localModels[j].getScore() - lastModel[j].getScore()) < 0.001))
                    continue iterations;
            }
            return true;
        }
        return false;
    }

    private boolean checkModels(Model[] localModels) {
        if (lastLocalModels.isEmpty()) return false;

        iterations: for (int i = lastLocalModels.size()-1; (i >= 0) && (lastLocalModels.size()-1-i < maxModelsSaved); i--) {
            Model[] lastModel = lastLocalModels.get(i);

            for (int j = 0; j < localModels.length; j++) {
                // Checking if the entire model is the same
                if (!localModels[j].equals(lastModel[j]))
                    continue iterations;
            }
            return true;
        }
        return false;
    }
}
