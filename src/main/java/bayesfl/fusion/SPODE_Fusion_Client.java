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
 * Package containing fusion methods for federated Bayesian networks.
 */
package bayesfl.fusion;

import bayesfl.model.Model;
import bayesfl.model.SPODE_Model;

/**
 * Client fusion of {@link SPODE_Model}s: the client takes the global parameters and,
 * with global tables, the global tables; otherwise its tables stay local and are
 * recomputed from its data when needed. The global arrays are shared, not copied:
 * the next local round copies the parameters before changing them.
 */
public class SPODE_Fusion_Client implements Fusion {

    @Override
    public Model fusion(Model localModel, Model globalModel) {
        SPODE_Model global = (SPODE_Model) globalModel;
        int n = localModel != null ? ((SPODE_Model) localModel).getNumInstances() : 0;
        return new SPODE_Model(global.structure, global.params, global.logTables, null, global.weighted,
                global.logPrior, n, global.getMaxIterations());
    }

    @Override
    public Model fusion(Model[] models) {
        throw new UnsupportedOperationException("Method not implemented");
    }
}
