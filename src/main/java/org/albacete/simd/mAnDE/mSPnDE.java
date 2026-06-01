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
 *    mSPnDE.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.mAnDE;

import java.util.Collection;
import java.util.Set;

import bayesfl.privacy.NumericNoiseGenerator;
import weka.core.Instance;

public interface mSPnDE {

    void buildTables(mAnDE model);

    /**
     * Adds per-cell DP noise to every count cell in this SPODE (globalCounts
     * and childCounts) and clamps the result to {@code >= 0} so that the
     * subsequent {@link #normalizeCounts} pass is well-defined. Must be
     * called after {@link #buildCountTables} and before the counts are sent
     * to the server (client-side DP).
     *
     * @param noise the noise generator (typically Laplace with sensitivity
     *              equal to the maximum per-row contribution of a single
     *              record, i.e. 2 for binary marginals).
     */
    void applyNoise(NumericNoiseGenerator noise);

    /**
     * Builds raw count tables (not normalized) from the local data.
     * Used in federated parameter learning to accumulate counts before fusion.
     *
     * @param model The mAnDE model containing the local data and metadata.
     */
    void buildCountTables(mAnDE model);

    /**
     * Adds the raw count tables from another SPODE (same structure) to this one.
     * Used by the server to aggregate counts from all clients.
     *
     * @param other The SPODE from another client whose counts are to be added.
     */
    void addCounts(mSPnDE other);

    /**
     * Normalizes the accumulated raw count tables into probability distributions.
     * Must be called after all clients' counts have been aggregated via addCounts.
     */
    void normalizeCounts();

    /**
     * Returns true if this SPODE already has its probability tables populated
     * (either from a local {@link #buildTables} pass or from a federated
     * {@link #normalizeCounts}). Used by {@link mAnDE#calculateTables_mSPnDEs}
     * to avoid overwriting federated parameters with locally-recomputed ones.
     */
    boolean hasProbTables();

    double[] probsForInstance(Instance inst, mAnDE model);

    void moreChildren(int child);

    void moreChildren(Collection<Integer> children);

    Set<Integer> getChildren();

    int getNChildren();

    mSPnDE copyDeep();

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
