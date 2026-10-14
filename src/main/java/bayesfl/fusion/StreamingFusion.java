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

/**
 * A server fusion that aggregates the local models one at a time, so the server
 * never holds every client's model at once. The {@code Server} then trains the
 * clients one after another and hands each local model over as soon as it is built.
 */
public interface StreamingFusion extends Fusion {

    /**
     * Starts the aggregation of a new round.
     */
    void begin();

    /**
     * Adds one local model to the aggregate.
     *
     * @param model The local model of a client.
     */
    void accumulate(Model model);

    /**
     * Ends the round.
     *
     * @return The global model.
     */
    Model finish();

    /**
     * Fusion of an array of models through the streaming aggregate.
     *
     * @param models The local models.
     * @return The global model.
     */
    @Override
    default Model fusion(Model[] models) {
        begin();
        for (Model m : models) accumulate(m);
        return finish();
    }
}
