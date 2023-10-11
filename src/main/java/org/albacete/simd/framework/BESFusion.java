package org.albacete.simd.framework;

import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.threads.BESThread;
import org.albacete.simd.utils.Problem;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import org.albacete.simd.threads.GESThread;
import static org.albacete.simd.utils.Utils.pdagToDag;

public class BESFusion extends FusionStage{
    
    ThreadStage besStage;
    
    public BESFusion(Problem problem, Graph currentGraph, ArrayList<Dag_n> graphs, BESStage besStage) {
        super(problem, currentGraph, graphs);
        this.besStage = besStage;
    }
    
    public boolean flag = false;

    @Override
    protected Dag_n fusion() {
        Dag_n fusionGraph = this.fusionIntersection();

        System.out.println("BES to obtain the fusion: ");

        Set<Edge> candidates = new HashSet<>();

        for (Edge e: this.currentGraph.getEdges()){
            if(fusionGraph.getEdge(e.getNode1(), e.getNode2())==null && fusionGraph.getEdge(e.getNode2(),e.getNode1())==null ) {
                candidates.add(Edges.directedEdge(e.getNode1(),e.getNode2()));
                candidates.add(Edges.directedEdge(e.getNode2(),e.getNode1()));
            }
        }

        BESThread fuse = new BESThread(this.problem,this.currentGraph,candidates);

        fuse.run();
        
        // We obtain the flag of the BES. If true, BESThread has improve the result.
        try {
            flag = fuse.getFlag();
        } catch (InterruptedException ex) {}
        
        // If the BESThread has not improved the previous result, we check if the fusion improves it.
        if (!flag) {
            double fusionScore = GESThread.scoreGraph(fusionGraph, problem);
            double currentScore = GESThread.scoreGraph(this.currentGraph, problem);
            
            if (fusionScore > currentScore) {
                flag = true;
                this.currentGraph = fusionGraph;
                return (Dag_n) this.currentGraph;
            } 
            
            // If the fusion doesn´t improves the result, we check if any previous BESThread has improved the results.
            else {
                GESThread thread = besStage.getMaxBDeuThread();
                System.out.println("thread"  + thread);
                if (thread.getScoreBDeu() != 0 && thread.getScoreBDeu() > currentScore) {
                    try {
                        this.currentGraph = thread.getCurrentGraph();
                        System.out.println(this.currentGraph);
                        flag = true;
                    } catch (InterruptedException ex) {
                        System.out.println("\n\n\n EXCEPCIÓN º\n\n\n");}
                    return (Dag_n) this.currentGraph;
                }
            }
        }
        
        try {
            this.currentGraph = fuse.getCurrentGraph();
            //System.out.println("Resultado del BES de la fusion: "+ BESThread.scoreGraph(this.currentGraph, problem));
            //this.currentGraph = Utils.removeInconsistencies(this.currentGraph);
            //System.out.println("Score Fusion sin inconsistencias: "+ BESThread.scoreGraph(this.currentGraph, problem));
        } catch (InterruptedException e1) {
            e1.printStackTrace();
        }

        pdagToDag(this.currentGraph);
        return new Dag_n(this.currentGraph);
        //return Utils.removeInconsistencies(this.currentGraph);
    }
}
