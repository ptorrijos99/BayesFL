package org.albacete.simd.framework;

import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;

import java.util.List;
import java.util.Set;

public abstract class BackwardStage extends ThreadStage{

    private static long sumTimeThreads = 0;
    private static long sumDeviationThreads = 0;
    private static int n = 0;

    public static double meanTimeTotal=0;
    public static double varianceTimeTotal =0;

    public BackwardStage(Problem problem, Graph currentGraph, int nThreads, int itInterleaving, List<Set<Edge>> subsets) {
        super(problem, currentGraph, nThreads, itInterleaving, subsets);
    }

    @Override
    protected void calculateStatsTimeTotal() {
        // Calculating mean
        for(GESThread g: gesThreads){
            n++;
            sumTimeThreads += g.getElapsedTime();
        }
        meanTimeTotal = (double) sumTimeThreads / n;

        // Calculating std
        for(GESThread g: gesThreads){
            sumDeviationThreads += Math.pow((g.getElapsedTime() - meanTimeTotal),2);
        }
        varianceTimeTotal = (double) sumDeviationThreads / n;
        System.out.println("Backwards Time Stats: n=" + n + " sumTimeThreads=" + sumTimeThreads + " meanTimeTotal: " +
                meanTimeTotal + " sumDeviationThreads=" + sumDeviationThreads + " varianceTimeTotal: " + varianceTimeTotal);
    }

}
