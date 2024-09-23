package consensusBN.Method;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Node;

import java.util.List;
import java.util.Random;

public class InitialDAGsWoRepeat_Method implements Population {

    private Random random;

    private int maxTreewidth;

    private Dag greedyDag;
    private double executionTimeGreedy;

    @Override
    public void initialize(List<Dag> dags, List<Node> alpha, List<Dag> alphaDags, int maxTreewidth, Random random) {
        this.random = random;
        this.maxTreewidth = maxTreewidth;
    }

    /**
     * Initialize the population. The initial population is composed of the edges of the original DAGs.
     * The edges of the DAGs are not repeated in the population.
     */
    @Override
    public boolean[][] initializePopulation(int populationSize) {
        return null;
    }

    @Override
    public Dag getUnionFromChromosome(boolean[] chromosome) {
        return null;
    }

    @Override
    public Dag getGreedyDag() {
        return null;
    }

    @Override
    public double getExecutionTimeGreedy() {
        return executionTimeGreedy;
    }
}
