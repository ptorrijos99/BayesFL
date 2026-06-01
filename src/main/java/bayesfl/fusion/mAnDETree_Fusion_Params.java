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
 *    mAnDETree_Fusion_Params.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.fusion;

import bayesfl.model.Model;
import bayesfl.model.mAnDETree;
import org.albacete.simd.mAnDE.mSPnDE;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Server-side fusion for the parameter-learning round of federated MiniAnDE.
 * <p>
 * Each client sends an {@link mAnDETree} containing raw count tables (built by
 * {@link bayesfl.algorithms.mAnDETree_LocalParams}). This class sums those
 * counts
 * across all clients and then normalizes the result into probability
 * distributions,
 * producing a single globally-consistent {@link mAnDETree} model.
 * </p>
 * <p>
 * The fusion preserves the SPODE structure (super-parents + child sets) from
 * Round 1
 * without any modification, and operates purely on the probability tables.
 * </p>
 */
public class mAnDETree_Fusion_Params implements Fusion {

    @Override
    public Model fusion(Model model1, Model model2) {
        return fusion(new Model[] { model1, model2 });
    }

    /**
     * Aggregates count tables from all client {@link mAnDETree} models.
     * <p>
     * For each SPODE key that exists in the first model (which defines the global
     * structure),
     * the counts from all other clients are added via {@link mSPnDE#addCounts}.
     * After
     * aggregation, each SPODE's counts are normalized to probability distributions
     * via
     * {@link mSPnDE#normalizeCounts}.
     * </p>
     *
     * @param models An array of local {@link mAnDETree} models from each client.
     * @return A new global {@link mAnDETree} with normalized probability tables.
     */
    @Override
    public Model fusion(Model[] models) {
        for (Model model : models) {
            if (!(model instanceof mAnDETree)) {
                throw new IllegalArgumentException(
                        "All models must be mAnDETree instances to use mAnDETree_Fusion_Params");
            }
        }

        // Use the first client's model as the accumulator (it already has count tables)
        mAnDETree reference = (mAnDETree) models[0];
        ConcurrentHashMap<Object, mSPnDE> fusedSPnDEs = reference.getModel();

        // Add counts from all remaining clients
        for (int i = 1; i < models.length; i++) {
            ConcurrentHashMap<Object, mSPnDE> clientSPnDEs = ((mAnDETree) models[i]).getModel();
            fusedSPnDEs.forEach((key, fusedSpode) -> {
                mSPnDE clientSpode = clientSPnDEs.get(key);
                if (clientSpode != null) {
                    fusedSpode.addCounts(clientSpode);
                }
            });
        }

        // Normalize all accumulated counts to probability distributions (in parallel)
        fusedSPnDEs.values().parallelStream().forEach(mSPnDE::normalizeCounts);

        return new mAnDETree(fusedSPnDEs, reference.getAlgorithm());
    }
}
