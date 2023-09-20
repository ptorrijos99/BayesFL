package org.albacete.simd.algorithms.bnbuilders;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import edu.cmu.tetrad.search.Fges2;
import org.albacete.simd.framework.*;

public class Fges_BNBuilder extends BNBuilder {
    
    public boolean setFaithfulnessAssumed;
    public boolean ges;

    public Fges_BNBuilder(DataSet data, boolean setFaithfulnessAssumed, boolean ges) {
        super(data, 1, -1, -1);
        this.setFaithfulnessAssumed = setFaithfulnessAssumed;
        this.ges = ges;
    }

    public Fges_BNBuilder(String path, boolean setFaithfulnessAssumed, boolean ges) {
        super(path, 1, -1, -1);
        this.setFaithfulnessAssumed = setFaithfulnessAssumed;
        this.ges = ges;
    }

    @Override
    public Graph search(){
        BDeuScore bdeu = new BDeuScore(this.getData());
        if (!ges) {
            Fges fges = new Fges(bdeu);
            fges.setFaithfulnessAssumed(setFaithfulnessAssumed);
            this.currentGraph = fges.search();
            this.score = fges.scoreDag(currentGraph);
        } else {
            Fges2 fges = new Fges2(bdeu);
            fges.setFaithfulnessAssumed(true);
            this.currentGraph = fges.search();
            this.score = fges.scoreDag(currentGraph);
        }

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
