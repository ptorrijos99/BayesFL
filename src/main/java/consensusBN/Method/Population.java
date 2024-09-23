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
}
