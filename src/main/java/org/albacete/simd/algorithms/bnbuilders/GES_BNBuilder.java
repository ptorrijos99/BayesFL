package org.albacete.simd.algorithms.bnbuilders;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.framework.BackwardStage;
import org.albacete.simd.framework.ForwardStage;
import org.albacete.simd.threads.BESThread;
import org.albacete.simd.threads.FESThread;
import org.albacete.simd.utils.Problem;
import org.albacete.simd.utils.Utils;

import java.util.LinkedList;
import java.util.Set;

public class GES_BNBuilder extends BNBuilder {

    private boolean fesFlag = false;
    private boolean besFlag = false;
    
    private final boolean parallel;


    public GES_BNBuilder(DataSet data, boolean parallel) {
        super(data, 1, Integer.MAX_VALUE, Integer.MAX_VALUE);
        super.setInitialGraph(new EdgeListGraph(new LinkedList<>(problem.getVariables())));
        this.parallel = parallel;
    }

    public GES_BNBuilder(String path, boolean parallel) {
        super(path, 1, Integer.MAX_VALUE, Integer.MAX_VALUE);
        super.setInitialGraph(new EdgeListGraph(new LinkedList<>(problem.getVariables())));
        this.parallel = parallel;
    }

    public GES_BNBuilder(Graph initialDag, DataSet data, boolean parallel) {
        this(data, parallel);
        super.setInitialGraph(new EdgeListGraph(initialDag));
        this.currentGraph = new EdgeListGraph(initialDag);
    }

    public GES_BNBuilder(Graph initialDag, String path, boolean parallel) {
        this(path, parallel);
        super.setInitialGraph(new EdgeListGraph(initialDag));
        this.currentGraph = new EdgeListGraph(initialDag);
    }

    public GES_BNBuilder(Graph initialDag, Problem problem, Set<Edge> subsetEdges, boolean parallel) {
        super(initialDag, problem, 1, Integer.MAX_VALUE,Integer.MAX_VALUE);
        super.setOfArcs = subsetEdges;
        super.setInitialGraph(new EdgeListGraph(initialDag));
        this.parallel = parallel;
    }

    @Override
    public boolean convergence() {
        // No changes in either fes or bes stages
        return !(fesFlag || besFlag);
    }

    @Override
    protected void initialConfig() {
    }

    @Override
    protected void repartition() {

    }

    @Override
    protected void forwardStage() throws InterruptedException {
        ForwardStage.meanTimeTotal = 0;
        FESThread fes = new FESThread(problem, super.getInitialGraph(), setOfArcs, this.nItInterleaving, false, false, parallel);
        fes.run();
        currentGraph = fes.getCurrentGraph();
        fesFlag = fes.getFlag();
        score = fes.getScoreBDeu();
    }

    @Override
    protected void forwardFusion() throws InterruptedException {

    }

    @Override
    protected void backwardStage() throws InterruptedException {
        BackwardStage.meanTimeTotal = 0;
        BESThread bes = new BESThread(problem, currentGraph, setOfArcs);
        bes.run();
        currentGraph = bes.getCurrentGraph();
        besFlag = bes.getFlag();
        score = bes.getScoreBDeu();
        currentGraph = Utils.removeInconsistencies(currentGraph);
    }

    @Override
    protected void backwardFusion() throws InterruptedException {

    }
    @Override
    public Graph search(){
        try {
            forwardStage();
            backwardStage();
        }catch(InterruptedException e){
            System.err.println("Interrupted Exception");
            e.printStackTrace();
        }
        return this.currentGraph;
    }
}
