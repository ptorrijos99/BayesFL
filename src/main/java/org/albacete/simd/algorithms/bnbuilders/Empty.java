package org.albacete.simd.algorithms.bnbuilders;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import java.util.LinkedList;
import org.albacete.simd.framework.*;

public class Empty extends BNBuilder {
    
    public Empty(DataSet data) {
        super(data, 1, -1, -1);
    }

    public Empty(String path) {
        super(path, 1, -1, -1);
    }

    @Override
    public Graph search(){
        BDeuScore bdeu = new BDeuScore(this.getData());
        Fges fges = new Fges(bdeu);
        currentGraph = new EdgeListGraph(new LinkedList<>(problem.getVariables()));
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
