package org.albacete.simd.algorithms.bnbuilders;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import org.albacete.simd.framework.*;

public class Fges_BNBuilder extends BNBuilder {
    
    public boolean setFaithfulnessAssumed;

    public Fges_BNBuilder(DataSet data, boolean setFaithfulnessAssumed) {
        super(data, 1, -1, -1);
        this.setFaithfulnessAssumed = setFaithfulnessAssumed;
    }

    public Fges_BNBuilder(String path, boolean setFaithfulnessAssumed) {
        super(path, 1, -1, -1);
        this.setFaithfulnessAssumed = setFaithfulnessAssumed;
    }

    @Override
    public Graph search(){
        BDeuScore bdeu = new BDeuScore(this.getData());

        Fges fges = new Fges(bdeu);
        fges.setFaithfulnessAssumed(setFaithfulnessAssumed);
        Graph initialGraph = super.getInitialGraph();
        if (initialGraph != null) {
            fges.setExternalGraph(initialGraph);
        }
        
        
        this.currentGraph = fges.search();
        this.score = fges.scoreDag(currentGraph);

        return this.currentGraph;
    }

    @Override
    protected boolean convergence() {
        return true;
    }

    @Override
    protected void initialConfig() {
    }

    @Override
    protected void repartition() {
    }

    @Override
    protected void forwardStage() throws InterruptedException {
    }

    @Override
    protected void forwardFusion() throws InterruptedException {
    }

    @Override
    protected void backwardStage() throws InterruptedException {
    }

    @Override
    protected void backwardFusion() throws InterruptedException {
    }

}
