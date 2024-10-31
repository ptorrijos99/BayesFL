package consensusBN.Method;

import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Node;

import java.util.*;

import static consensusBN.BetaToAlpha.transformToAlpha;
import static consensusBN.ConsensusUnion.*;

public class InitialDAGs_Method implements Population {

    private Random random;
    private int maxTreewidth;

    private List<Node> alpha;
    private List<Edge>[] edgesOriginal;
    private int nDags;
    private int[] nEdges;
    private int totalEdges;
    private int[] firstIndex;

    private Dag greedyDag;
    private List<Dag> greedyDags;
    private double executionTimeGreedy;

    public Boolean useMinCut;
    public MinCutTreeWidthUnion minCutTreeWidthUnion;

    public InitialDAGs_Method() {
        this.useMinCut = false;
    }

    public InitialDAGs_Method(Boolean useMinCut) {
        this.useMinCut = useMinCut;
    }

    @Override
    public void initialize(List<Dag> dags, List<Node> alpha, List<Dag> alphaDags, int maxTreewidth, Random random) {
        this.alpha = alpha;
        this.random = random;
        this.maxTreewidth = maxTreewidth;

        nDags = dags.size();
        nEdges = new int[nDags];
        this.edgesOriginal = new ArrayList[nDags];
        this.firstIndex = new int[nDags];
        for (int i = 0; i < nDags; i++) {
            nEdges[i] = dags.get(i).getNumEdges();
            edgesOriginal[i] = new ArrayList<>(dags.get(i).getEdges());
            totalEdges += nEdges[i];
            if (i == 0) {
                firstIndex[i] = 0;
            } else {
                firstIndex[i] = firstIndex[i-1] + nEdges[i-1];
            }
        }

        if (useMinCut && maxTreewidth > 1) {
            minCutTreeWidthUnion = new MinCutTreeWidthUnion(dags, 10, maxTreewidth);
            minCutTreeWidthUnion.experiments = true;
            minCutTreeWidthUnion.fusion();

            int size = minCutTreeWidthUnion.outputExperimentDAGsList.size();
            greedyDags = minCutTreeWidthUnion.outputExperimentDAGsList.get(size-1);
            greedyDag = minCutTreeWidthUnion.outputExperimentDAGs.get(size-1);
            executionTimeGreedy = minCutTreeWidthUnion.outputExperimentTimes.get(size-1);
        } else {
            double startTime = System.currentTimeMillis();
            greedyDags = originalDAGsGreedyTreewidthBefore(dags, alpha, "" + maxTreewidth);
            greedyDag = fusionUnion(greedyDags);
            executionTimeGreedy = (System.currentTimeMillis() - startTime) / 1000;
        }
    }

    /**
     * Initialize the population. The initial population is composed of the edges of the original DAGs.
     */
    @Override
    public boolean[][] initializePopulation(int populationSize) {
        boolean[][] population = new boolean[populationSize][totalEdges];

        // Add the greedy solution to the population
        for (int i = 0; i < nDags; i++) {
            for (Edge edge : greedyDags.get(i).getEdges()) {
                population[0][edgesOriginal[i].indexOf(edge) + firstIndex[i]] = true;
            }
        }

        // Add the greedy solution with maxTreewidth-1 to the population
        List<Dag> greedyDagsMaxTreewidthMinusOne;
        if (useMinCut && maxTreewidth > 2) {
            System.out.println("tw = " + maxTreewidth);
            System.out.println("Using MinCut");
            System.out.println("Size: " + minCutTreeWidthUnion.outputExperimentDAGsList.size());
            int size = minCutTreeWidthUnion.outputExperimentDAGsList.size();
            greedyDagsMaxTreewidthMinusOne = minCutTreeWidthUnion.outputExperimentDAGsList.get(size-2);
        } else {
            greedyDagsMaxTreewidthMinusOne = originalDAGsGreedyTreewidthBefore(greedyDags, alpha, ""+(maxTreewidth-1));
        }

        for (int i = 0; i < nDags; i++) {
            for (Edge edge : greedyDagsMaxTreewidthMinusOne.get(i).getEdges()) {
                population[1][edgesOriginal[i].indexOf(edge) + firstIndex[i]] = true;
            }
        }
        // Initialize the rest of the population with random individuals
        for (int i = 2; i < populationSize; i++) {
            for (int j = 2; j < totalEdges; j++) {
                population[i][j] = random.nextBoolean();
            }
        }

        return population;
    }
    
    @Override
    public Dag getUnionFromChromosome(boolean[] chromosome) {
        // Create the DAG that corresponds to each individual
        ArrayList<Dag> candidates = fromChromosomeToDags(chromosome);

        ArrayList<Dag> outputDags = new ArrayList<>();
        for (Dag dag : candidates) {
            outputDags.add(transformToAlpha(dag, alpha));
        }

        return applyUnion(alpha, outputDags);
    }

    private ArrayList<Dag> fromChromosomeToDags(boolean[] chromosome) {
        ArrayList<Dag> dags = new ArrayList<>();
        int cumulativeEdges = 0;
        for (int i = 0; i < nDags; i++) {
            Dag dag = new Dag(alpha);
            for (int j = 0; j < nEdges[i]; j++) {
                if (chromosome[cumulativeEdges]) {
                    dag.addEdge(edgesOriginal[i].get(j));
                }
                cumulativeEdges++;
            }
            dags.add(dag);
        }

        return dags;
    }

    @Override
    public Dag getGreedyDag() {
        return greedyDag;
    }

    @Override
    public double getExecutionTimeGreedy() {
        return executionTimeGreedy;
    }
}
