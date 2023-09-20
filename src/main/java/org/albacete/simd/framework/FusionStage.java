package org.albacete.simd.framework;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Problem;

import java.util.ArrayList;

public abstract class FusionStage extends Stage {


    public FusionStage(Problem problem, Graph currentGraph, ArrayList<Dag_n> graphs) {
        super(problem, currentGraph);
        this.graphs = graphs;
    }

    @Override
    public boolean run() throws InterruptedException {
        currentGraph = fusion();
        return true;
    }

    protected abstract Dag_n fusion() throws InterruptedException;


    protected Dag_n fusionIntersection(){
        ArrayList<Node> order = new ArrayList<>(this.currentGraph.getCausalOrdering()); // currentGraph.getCausalOrdering
        for(int i = 0; i < this.graphs.size(); i++) {
            for(Edge e:this.graphs.get(i).getEdges()) {
                if((order.indexOf(e.getNode1()) < order.indexOf(e.getNode2())) && (e.getEndpoint1()== Endpoint.TAIL && e.getEndpoint2()==Endpoint.ARROW))
                    continue;

                if((order.indexOf(e.getNode1()) > order.indexOf(e.getNode2())) && (e.getEndpoint1()== Endpoint.ARROW && e.getEndpoint2()==Endpoint.TAIL))
                    continue;

                if(e.getEndpoint1()==Endpoint.TAIL)
                    e.setEndpoint1(Endpoint.ARROW);
                else
                    e.setEndpoint1(Endpoint.TAIL);

                if(e.getEndpoint2()==Endpoint.TAIL)
                    e.setEndpoint2(Endpoint.ARROW);
                else
                    e.setEndpoint2(Endpoint.TAIL);

            }

        }
        Graph graph = new EdgeListGraph_n(this.currentGraph);
        // Looping over each edge of the currentGraph and checking if it has been deleted in any of the resulting graphs of the BES stage.
        // If it has been deleted, then it is removed from the currentGraph.
        for(Edge e: graph.getEdges()) {
            for(Dag_n g: this.graphs)
                if(!g.containsEdge(e)) {
                    graph.removeEdge(e);
                    break;
                }
        }
        return new Dag_n(graph);
    }

}
