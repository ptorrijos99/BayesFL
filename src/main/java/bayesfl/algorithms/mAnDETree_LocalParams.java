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
 *    mAnDETree_LocalParams.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.algorithms;

import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.model.Model;
import bayesfl.model.mAnDETree;
import bayesfl.privacy.NumericNoiseGenerator;
import org.albacete.simd.mAnDE.mAnDE;
import org.albacete.simd.mAnDE.mSPnDE;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.concurrent.ConcurrentHashMap;

/**
 * A local algorithm for the parameter-learning round of federated MiniAnDE.
 * <p>
 * It receives the globally-fused {@link mAnDETree} structure (containing which
 * SPODEs
 * exist and which children each SPODE has), and counts raw instances from the
 * local
 * client data into count tables — without normalizing them. The resulting model
 * is ready to be aggregated by {@link bayesfl.fusion.mAnDETree_Fusion_Params}.
 * </p>
 * <p>
 * This avoids using Weka's {@code NaiveBayes} (which allocates one
 * {@code DiscreteEstimator} per attribute per class, exploding memory for
 * high-dimensional datasets) and preserves the child-set pruning from
 * the MiniAnDE structure-learning phase.
 * </p>
 */
public class mAnDETree_LocalParams implements LocalAlgorithm {

    /**
     * The globally-fused MiniAnDE structure: contains which SPODEs exist and
     * which children each SPODE has. Children sets come from the union of all
     * clients' RF-extracted structures.
     */
    private final mAnDETree globalStructure;

    /**
     * Optional pre-computed cut points from federated discretization.
     * If not null, data will be discretized before counting.
     */
    private final double[][] cutPoints;

    /**
     * Optional DP noise generator applied to each cell of the local count
     * tables before they are sent to the server. {@code null} = no DP.
     */
    private final NumericNoiseGenerator noiseGenerator;

    /**
     * Constructor.
     *
     * @param globalStructure The globally-fused mAnDETree structure from Round 1.
     * @param cutPoints       Optional federated discretization cut points (null =
     *                        already discretized).
     */
    public mAnDETree_LocalParams(mAnDETree globalStructure, double[][] cutPoints) {
        this(globalStructure, cutPoints, null);
    }

    /**
     * Constructor with DP noise.
     *
     * @param globalStructure The globally-fused mAnDETree structure from Round 1.
     * @param cutPoints       Optional federated discretization cut points (null =
     *                        already discretized).
     * @param noiseGenerator  Optional client-side DP noise generator. If not
     *                        null, noise is added to the local count tables
     *                        before they are returned, i.e. client-side
     *                        privatization for the parameter-federation round.
     */
    public mAnDETree_LocalParams(mAnDETree globalStructure, double[][] cutPoints,
                                  NumericNoiseGenerator noiseGenerator) {
        this.globalStructure = globalStructure;
        this.cutPoints = cutPoints;
        this.noiseGenerator = noiseGenerator;
    }

    /**
     * Builds a local model by counting instances from the client's local data
     * into the globally-shared SPODE structure.
     *
     * @param data The local client data.
     * @return An {@link mAnDETree} containing raw count tables (not yet
     *         normalized).
     */
    @Override
    public Model buildLocalModel(Data data) {
        Instances instances = prepareInstances(data);
        mAnDE model = prepareModel(instances);

        // Deep-copy the global structure so that each client starts from clean count
        // tables
        ConcurrentHashMap<Object, mSPnDE> localSPnDEs = new ConcurrentHashMap<>();
        globalStructure.getModel().forEach((key, spode) -> localSPnDEs.put(key, spode.copyDeep()));

        // Build raw count tables for each SPODE in parallel
        model.mSPnDEs = localSPnDEs;
        localSPnDEs.values().parallelStream().forEach(spode -> spode.buildCountTables(model));

        // Client-side DP: privatize each count cell before sharing with the server.
        if (noiseGenerator != null) {
            localSPnDEs.values().parallelStream().forEach(spode -> spode.applyNoise(noiseGenerator));
        }

        return new mAnDETree(localSPnDEs, globalStructure.getAlgorithm());
    }

    /**
     * Memory-efficient accumulation: counts this client's data DIRECTLY into the
     * server-side {@code accumulator}, one SPODE at a time, WITHOUT materializing
     * a full second copy of the (potentially huge) global count-table model.
     * <p>
     * For each SPODE we take a structure-only {@link mSPnDE#copyDeep() copy},
     * build only that SPODE's count table on this client's data, optionally
     * privatize it (client-side DP, same semantics as
     * {@link #buildLocalModel}+{@code addCounts}), add it into the accumulator's
     * matching SPODE, and let the temporary be garbage-collected. Peak extra
     * memory is therefore one SPODE's count table per worker thread instead of a
     * whole second global model — which is what lets high-dimensional A2DE runs
     * fit in memory as the number of clients K (and thus the union structure)
     * grows.
     * <p>
     * The {@code accumulator} must already hold allocated count tables (e.g. from
     * the first client's {@link #buildLocalModel}). Each {@code accSpode} is a
     * distinct object handled by a single worker, so the parallel add is safe.
     *
     * @param accumulator server-side running counts (same structure as the global
     *                    model); MUTATED in place.
     * @param data        the local client data.
     */
    public void accumulateInto(ConcurrentHashMap<Object, mSPnDE> accumulator, Data data) {
        Instances instances = prepareInstances(data);
        final mAnDE model = prepareModel(instances);

        accumulator.values().parallelStream().forEach(accSpode -> {
            mSPnDE temp = accSpode.copyDeep();        // structure only (no count arrays)
            temp.buildCountTables(model);             // counts for THIS client only
            if (noiseGenerator != null) {
                temp.applyNoise(noiseGenerator);      // client-side DP on its own counts
            }
            accSpode.addCounts(temp);                 // fold into the accumulator
        });
    }

    /**
     * Casts the client data to {@link Instances} and applies the optional
     * pre-computed discretization cut points.
     */
    private Instances prepareInstances(Data data) {
        if (!(data instanceof Weka_Instances)) {
            throw new IllegalArgumentException("Data must be a Weka_Instances object");
        }
        Instances instances = (Instances) data.getData();

        // Apply pre-computed cut points if data has not been pre-discretized
        if (this.cutPoints != null) {
            Dummy filter = new Dummy();
            filter.setCutPoints(this.cutPoints);
            try {
                filter.setInputFormat(instances);
                instances = Filter.useFilter(instances, filter);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return instances;
    }

    /**
     * Prepares the shared mAnDE shell with this client's data and the schema
     * metadata required by {@link mSPnDE#buildCountTables}.
     */
    private mAnDE prepareModel(Instances instances) {
        mAnDE model = globalStructure.getAlgorithm().getAlgorithm();
        try {
            model.checkData(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Initialize metadata fields not set by checkData() but required by buildCountTables()
        model.initMetadata();
        return model;
    }

    /**
     * Delegates to {@link #buildLocalModel(Data)} — no iterative refinement needed.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        return buildLocalModel(data);
    }

    /**
     * No refinement step for this algorithm.
     */
    @Override
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return localModel;
    }

    @Override
    public String getAlgorithmName() {
        return "mAnDETree_LocalParams";
    }

    @Override
    public String getRefinementName() {
        return "None";
    }
}
