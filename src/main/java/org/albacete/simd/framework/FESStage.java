package org.albacete.simd.framework;

import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.threads.FESThread;
import org.albacete.simd.utils.Problem;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class FESStage extends ForwardStage{
    
    private boolean speedUp;
    private boolean update;
    private boolean parallel;

    public FESStage(Problem problem, int nThreads, int itInterleaving, List<Set<Edge>> subsets, boolean speedUp, boolean update, boolean parallel) {
        super(problem, nThreads, itInterleaving, subsets);
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    public FESStage(Problem problem, Graph currentGraph, int nThreads, int itInterleaving, List<Set<Edge>> subsets, boolean speedUp, boolean update, boolean parallel) {
        super(problem, currentGraph, nThreads, itInterleaving, subsets);
        this.speedUp = speedUp;
        this.update = update;
        this.parallel = parallel;
    }

    @Override
    public boolean run() {
        config();
        try {
            runThreads();
            flag = checkWorkingStatus();
            return flag;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return false;
        }
    }

    private void config() {
        // Initializing Graphs structure
        this.graphs = new ArrayList<>();

        // Rebuilding hashIndex
        //problem.buildIndexing(currentGraph);

        // Creating each ThFES runnable
        if (this.currentGraph == null) {
            for (int i = 0; i < this.nThreads; i++) {
                this.gesThreads[i] = new FESThread(this.problem, this.subsets.get(i), this.itInterleaving, speedUp, update, parallel);
            }
        }
        else{
            for (int i = 0; i < this.nThreads; i++) {
                this.gesThreads[i] = new FESThread(this.problem, this.currentGraph, this.subsets.get(i), this.itInterleaving, speedUp,  update, parallel);
            }
        }

        // Initializing thread config
        for(int i = 0 ; i< this.nThreads; i++){
            // Resetting the search flag
            this.gesThreads[i].resetFlag();
            this.threads[i] = new Thread(this.gesThreads[i]);
        }
    }
}
