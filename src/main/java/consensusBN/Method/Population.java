package consensusBN.Method;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Node;

import java.util.List;
import java.util.Random;

public interface Population {

    public void initialize(List<Dag> dags, List<Node> alpha, List<Dag> alphaDags, int maxTreewidth, Random random);

    public boolean[][] initializePopulation(int populationSize);

    public Dag getUnionFromChromosome(boolean[] chromosome);

    public Dag getGreedyDag();

    public double getExecutionTimeGreedy();

    /** Controls whether the greedy solution is injected into population[0]/[1] as warmstart.
     *  Default no-op — implementing classes override this to set their internal flag. */
    default void setUseGreedyWarmstart(boolean use) { }
}
