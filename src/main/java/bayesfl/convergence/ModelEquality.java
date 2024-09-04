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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This convergence criterion checks if the local models have converged by checking if all the models of this iteration
 * have appeared in any of the previous iterations.
 */
public class ModelEquality implements Convergence {

    private List<Set<Model>> lastLocalModels;

    /**
     * Checks if the local models have converged by checking if all the models of this iteration
     * have appeared in any of the previous iterations.
     *
     * @param localModels The local models of the clients.
     * @return True if the local models have converged, false otherwise.
     */
    @Override
    public boolean checkConvergence(Model[] localModels) {
        boolean convergence = true;

        if (lastLocalModels == null) {
            lastLocalModels = new ArrayList<>(localModels.length);
            for (int i = 0; i < localModels.length; i++) {
                lastLocalModels.add(new HashSet<>());
            }
        }

        for (int i = 0; i < localModels.length; i++) {
            convergence = lastLocalModels.get(i).contains(localModels[i]);
            if (!convergence) {
                break;
            }
        }

        Model[] clones = localModels.clone();
        for (int i = 0; i < clones.length; i++) {
            lastLocalModels.get(i).add(clones[i]);
        }

        return convergence;
    }
}
