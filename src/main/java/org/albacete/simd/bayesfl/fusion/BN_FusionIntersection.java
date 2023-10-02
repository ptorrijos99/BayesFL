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
 *    BN_FusionIntersection.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.bayesfl.fusion;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.bayesfl.model.BN;
import org.albacete.simd.bayesfl.model.Model;

import java.util.ArrayList;

public class BN_FusionIntersection implements Fusion {

    @Override
    public Model fusion(Model model1, Model model2) {
        if (!(model1 instanceof BN) || !(model2 instanceof BN)) {
            throw new IllegalArgumentException("The models must be objects of the BN class to use BN_FusionIntersection");
        }

        Dag_n[] graphs = new Dag_n[2];
        graphs[0] = ((BN) model1).getModel();
        graphs[1] = ((BN) model2).getModel();

        return new BN(fusionIntersection(graphs));
    }

    @Override
    public Model fusion(Model[] models) {
        for (Model model : models) {
            if (!(model instanceof BN)) {
                throw new IllegalArgumentException("The models must be objects of the BN class to use BN_FusionIntersection");
            }
        }

        Dag_n[] graphs = new Dag_n[models.length];
        for (int i = 0; i < models.length; i++) {
            graphs[i] = ((BN) models[i]).getModel();
        }

        return new BN(fusionIntersection(graphs));
    }

    private Dag_n fusionIntersection(Dag_n[] graphs) {
        ArrayList<Node> order = new ArrayList<>(graphs[0].getCausalOrdering()); // Randomly first graph

        for(Dag_n graph : graphs) {
            for(Edge e : graph.getEdges()) {
                if((order.indexOf(e.getNode1()) < order.indexOf(e.getNode2())) &&
                        (e.getEndpoint1() == Endpoint.TAIL && e.getEndpoint2() == Endpoint.ARROW))
                    continue;

                if((order.indexOf(e.getNode1()) > order.indexOf(e.getNode2())) &&
                        (e.getEndpoint1() == Endpoint.ARROW && e.getEndpoint2() == Endpoint.TAIL))
                    continue;

                if(e.getEndpoint1() == Endpoint.TAIL)
                    e.setEndpoint1(Endpoint.ARROW);
                else
                    e.setEndpoint1(Endpoint.TAIL);

                if(e.getEndpoint2() == Endpoint.TAIL)
                    e.setEndpoint2(Endpoint.ARROW);
                else
                    e.setEndpoint2(Endpoint.TAIL);

            }

        }
        Graph graph = new EdgeListGraph_n(graphs[0]);
        // Looping over each edge of the first graph and checking if it has been deleted in any of the resulting graphs of the BES stage.
        // If it has been deleted, then it is removed from the currentGraph.
        for(Edge e: graph.getEdges()) {
            for(Dag_n g: graphs)
                if(!g.containsEdge(e)) {
                    graph.removeEdge(e);
                    break;
                }
        }
        return new Dag_n(graph);
    }
}
