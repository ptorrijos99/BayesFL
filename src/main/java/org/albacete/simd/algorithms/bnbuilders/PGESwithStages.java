package org.albacete.simd.algorithms.bnbuilders;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.clustering.Clustering;
import org.albacete.simd.framework.*;
import org.albacete.simd.threads.GESThread;

public class PGESwithStages extends BNBuilder {

    private boolean fesFlag = false;
    private boolean besFlag = false;
    
    private FESStage fesStage;
    private BESStage besStage;
    
    private final boolean speedUp;
    private final boolean update;
    private final boolean parallel;

    private final Clustering clustering;

    public PGESwithStages(DataSet data, Clustering clustering, int nThreads, int maxIterations, int nItInterleaving, boolean speedUp, boolean update, boolean parallel) {

        super(data, nThreads, maxIterations, nItInterleaving);
        this.clustering = clustering;
        this.clustering.setProblem(super.getProblem());
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    public PGESwithStages(String path, Clustering clustering, int nThreads, int maxIterations, int nItInterleaving, boolean speedUp, boolean update, boolean parallel) {
        super(path, nThreads, maxIterations, nItInterleaving);
        this.clustering = clustering;
        this.clustering.setProblem(super.getProblem());
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    public PGESwithStages(Graph initialGraph, String path, Clustering clustering, int nThreads, int maxIterations, int nItInterleaving, boolean speedUp, boolean update, boolean parallel) {
        super(initialGraph, path, nThreads, maxIterations, nItInterleaving);
        this.clustering = clustering;
        this.clustering.setProblem(super.getProblem());
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    public PGESwithStages(Graph initialGraph, DataSet data, Clustering clustering, int nThreads, int maxIterations, int nItInterleaving, boolean speedUp, boolean update, boolean parallel) {
        super(initialGraph, data, nThreads, maxIterations, nItInterleaving);
        this.clustering = clustering;
        this.clustering.setProblem(super.getProblem());
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    @Override
    protected boolean convergence() {
        // Checking Iterations
        if (it >= this.maxIterations)
            return true;

        it++;
        System.out.println("\n\nIterations: " + it);

        // Checking working status
        /*if(!fesFlag && !besFlag){
            return true;
        }*/
        double currentScore = GESThread.scoreGraph(this.currentGraph, this.problem);

        System.out.println("Current: " + currentScore + ", prev: "+ prevScore);
        if(currentScore > prevScore){

            prevScore = currentScore;
            return false;
        }
        else{
            return true;
        }
    }

    @Override
    protected void initialConfig() {

    }

    @Override
    protected void repartition() {
        this.subSets = clustering.generateEdgeDistribution(nThreads);
    }

    @Override
    protected void forwardStage(){
        fesStage = new FESStage(problem, currentGraph,nThreads,nItInterleaving, subSets, speedUp, update, parallel);
        fesFlag = fesStage.run();
        graphs = fesStage.getGraphs();
    }

    @Override
    protected void forwardFusion() throws InterruptedException {
        FESFusion fesFusion = new FESFusion(problem, currentGraph, graphs, update);
        fesFusion.run();
        fesFlag = fesFusion.flag;
        currentGraph = fesFusion.getCurrentGraph();
    }

    @Override
    protected void backwardStage(){
        besStage = new BESStage(problem, currentGraph, nThreads, nItInterleaving, subSets);
        besFlag = besStage.run();
        graphs = besStage.getGraphs();
    }

    @Override
    protected void backwardFusion() throws InterruptedException {
        BESFusion besFusion = new BESFusion(problem, currentGraph, graphs, besStage);
        besFusion.run();
        besFlag = besFusion.flag;
        currentGraph = besFusion.getCurrentGraph();
    }
/*
    public static void main(String[] args){
        // 1. Read Data
        String path = "src/test/resources/alarm.xbif_.csv";
        DataSet ds = Utils.readData(path);

        // 2. Configuring algorithm
        PGESwithStages pGESv2= new PGESwithStages(ds, 2, 15, 5);

        // 3. Running Algorithm
        pGESv2.search();

        // 4. Printing out the results
        System.out.println("Number of Iterations: " + pGESv2.getIterations());
        System.out.println("Resulting Graph: " + pGESv2.getCurrentGraph());


    }
 */
}
