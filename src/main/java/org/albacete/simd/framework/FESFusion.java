package org.albacete.simd.framework;

import consensusBN.ConsensusUnion;
import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.SearchGraphUtils;
import org.albacete.simd.threads.FESThread;
import org.albacete.simd.utils.Problem;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Utils;

import static org.albacete.simd.utils.Utils.pdagToDag;

public class FESFusion extends FusionStage{



    public FESFusion(Problem problem, Graph currentGraph, ArrayList<Dag_n> graphs, boolean update) {
        super(problem, currentGraph, graphs);
        this.update = update;
    }
    
    public boolean flag = false;
    private final boolean update;

    @Override
    public Dag_n fusion() {
        // Applying ConsensusUnion fusion
        ConsensusUnion fusion = new ConsensusUnion(this.graphs);
        Graph fusionGraph = fusion.union();

        // Applying FES to the fusion graph
        if (currentGraph == null) {
            flag = true;
            this.currentGraph = new EdgeListGraph(new LinkedList<>(fusionGraph.getNodes()));
        }
        System.out.println("FES to obtain the fusion: ");
        

        Set<Edge> candidates = new HashSet<>();
        
        
        for (Edge e : fusionGraph.getEdges()) {
            if (this.currentGraph.getEdge(e.getNode1(), e.getNode2()) != null || this.currentGraph.getEdge(e.getNode2(), e.getNode1()) != null)
                continue;
            candidates.add(Edges.directedEdge(e.getNode1(), e.getNode2()));
            candidates.add(Edges.directedEdge(e.getNode2(), e.getNode1()));
        }
        

        FESThread fuse = new FESThread(this.problem,this.currentGraph,candidates,candidates.size(),false, update,true);

        fuse.run();
        
        // We obtain the flag of the FES. If true, FESThread has improve the result.
        try {
            flag = flag || fuse.getFlag();
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
        
        // If the FESThread has not improved the previous result, we check if the fusion improves it.
        if (!flag) {
            double fusionScore = GESThread.scoreGraph(fusionGraph, problem);
            double currentScore = GESThread.scoreGraph(this.currentGraph, problem);
            
            if (fusionScore > currentScore) {
                flag = true;
                this.currentGraph = fusionGraph;
                System.out.println("  FESFusion -> FUSION, " + fusionScore);
                return (Dag_n) this.currentGraph;
            } 
            /*
            // If the fusion doesn´t improves the result, we check if any previous FESThread has improved the results.
            else {
                GESThread thread = fesStage.getMaxBDeuThread();
                if (thread.getScoreBDeu() > currentScore) {
                    try {
                        this.currentGraph = thread.getCurrentGraph();
                        flag = true;
                    } catch (InterruptedException ex) {}
                    System.out.println("  FESFusion -> THREAD, " + thread.getScoreBDeu());
                    this.currentGraph = new Dag_n(this.currentGraph);
                    return (Dag_n) this.currentGraph;
                }
            }*/
        }
        
        try {
            this.currentGraph = fuse.getCurrentGraph();
        } catch (InterruptedException e1) {
            e1.printStackTrace();
        }

        //pdagToDag(this.currentGraph);
        this.currentGraph = Utils.removeInconsistencies(this.currentGraph);
        //this.currentGraph = SearchGraphUtils.dagFromCPDAG(this.currentGraph);
        return new Dag_n(this.currentGraph);

        //return Utils.removeInconsistencies(this.currentGraph);
    }
}
