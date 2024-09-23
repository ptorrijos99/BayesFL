package consensusBN;

import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import org.albacete.simd.utils.Utils;

import java.util.*;
import java.util.stream.IntStream;

import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;
import static consensusBN.ConsensusUnion.*;

public class GeneticTreeWidthUnion {

    // Parameters
    public int numIterations = 1000;
    public int populationSize = 20;
    public int maxTreewidth = 5;
    public Boolean candidatesFromInitialDAGs = false;

    // Best values and stats
    private boolean[] bestIndividual;
    public Dag bestDag;
    public double bestFitness = Double.MAX_VALUE;
    public double executionTime;
    public double executionTimeUnion;
    public double executionTimeGreedy;

    // "Final" variables
    private final Random random;
    private int totalEdges;
    private final int nDags;
    private final int[] nEdges;
    private HashMap<Edge, Integer> edgeFrequency;
    private ArrayList<Integer> edgeFrequencyArray;
    private final ArrayList<Node> alpha;
    private final ArrayList<Dag> alphaDags;
    private final ArrayList<Edge> edgesOriginal;
    private ArrayList<Edge> edgesAlpha;
    private HashMap<Edge, Integer> edgePosition;
    public Dag greedyDag;
    public Dag fusionUnion;

    // Variables
    private boolean[][] population;
    private double[] fitness;
    private int[] treeWidths;

    public GeneticTreeWidthUnion(List<Dag> dags, int seed) {
        this.random = new Random(seed);
        this.nDags = dags.size();
        this.nEdges = new int[nDags];
        this.edgesOriginal = new ArrayList<>();
        for (int i = 0; i < nDags; i++) {
            nEdges[i] = dags.get(i).getNumEdges();
            edgesOriginal.addAll(dags.get(i).getEdges());
        }

        // Transform the DAGs to the same alpha order
        alpha = alphaOrder(dags);
        alphaDags = new ArrayList<>();
        for (Dag dag : dags) {
            alphaDags.add(transformToAlpha(dag, alpha));
        }
        System.out.println("Alpha order: " + alpha);

        // Apply the union of the DAGs
        double startTime = System.currentTimeMillis();
        fusionUnion = applyUnion(alpha, alphaDags);
        executionTimeUnion = (System.currentTimeMillis() - startTime) / 1000;
    }

    public GeneticTreeWidthUnion(List<Dag> dags, int seed, int maxTreewidth) {
        this(dags, seed);
        this.maxTreewidth = maxTreewidth;
    }

    /**
     * Complete union of the DAGs limiting the tree-width of the fused DAG.
     * @return The union of the DAGs.
     */
    public Dag fusionUnion() {
        double startTime = System.currentTimeMillis();

        initializeVars();

        if (candidatesFromInitialDAGs) {
            initializeInitialDAGs();
        }
        else {
            initializeFusionDAGs();
        }

        for (int i = 0; i < numIterations; i++) {
            //System.out.println("Iteration " + j);
            evaluate();
            crossover();
            evaluate();
            mutate();
        }

        executionTime = (System.currentTimeMillis() - startTime) / 1000;

        return bestDag;
    }

    private void initializeVars() {
        treeWidths = new int[populationSize];

        // Order the edges of all the DAGs by the number of times they appear
        int k = 0;
        edgeFrequency = new HashMap<>();
        edgePosition = new HashMap<>();
        edgesAlpha = new ArrayList<>();
        edgeFrequencyArray = new ArrayList<>();
        for (Dag d : alphaDags) {
            for (Edge edge : d.getEdges()) {
                // Add the edge to the map. If it already exists, increase its frequency
                Integer added = edgeFrequency.put(edge, edgeFrequency.getOrDefault(edge, 0) + 1);
                // If the edge is new, add it to the list of edges and the map with its position
                if (added == null) {
                    edgesAlpha.add(edge);
                    edgePosition.put(edge, k);
                    edgeFrequencyArray.add(1);
                    k++;
                }
            }
        }

        double startTime = System.currentTimeMillis();
        greedyDag = applyGreedyMaxTreewidth(alpha, edgesAlpha, ""+maxTreewidth);
        executionTimeGreedy = (System.currentTimeMillis() - startTime) / 1000;
    }

    private void initializeFusionDAGs() {
        totalEdges = edgeFrequency.size();
        int maxFreq = Collections.max(edgeFrequency.values());
        int minFreq = Collections.min(edgeFrequency.values()) - 1;

        // Initialize the population
        population = new boolean[populationSize][totalEdges];
        boolean uniform = minFreq == maxFreq;

        // Add the greedy solution to the population
        for (int i = 0; i < totalEdges; i++) {
            population[0][i] = greedyDag.containsEdge(edgesAlpha.get(i));
        }

        // Add the greedy solutions with maxTreewidth-1 to the population
        Dag greedy = applyGreedyMaxTreewidth(alpha, edgesAlpha, ""+(maxTreewidth-1));
        for (int i = 0; i < totalEdges; i++) {
            population[1][i] = greedy.containsEdge(edgesAlpha.get(i));
        }

        // Initialize the rest of the population with random individuals based on the frequency of the edges
        for (int i = 2; i < populationSize; i++) {
            for (int j = 0; j < totalEdges; j++) {
                if (uniform) {
                    population[i][j] = random.nextBoolean();
                } else {
                    double normalized = (double) (edgeFrequencyArray.get(j) - minFreq) / (maxFreq - minFreq);
                    population[i][j] = random.nextDouble() < 1/(1-Math.log(normalized));
                }
            }
        }

        bestIndividual = population[0].clone();
    }

    private void initializeInitialDAGs() {
        totalEdges = edgesOriginal.size();
        population = new boolean[populationSize][totalEdges];

        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < totalEdges; j++) {
                population[i][j] = random.nextBoolean();
            }
        }

        bestIndividual = population[0].clone();
    }

    private void evaluate() {
        fitness = new double[populationSize];

        IntStream.range(0, populationSize)
                .parallel()
                .forEach(i -> fitness[i] = calculateFitness(i, 0));

        /*System.out.println("Fitness: ");
        for (int i = 0; i < populationSize; i++) {
            System.out.printf("%.2f ", fitness[i]);
        }
        System.out.println();*/
    }

    private void crossover() {
        boolean[][] newPopulation = new boolean[populationSize][totalEdges];
        // Add the best global individual to the new population
        //System.out.println("Best individual: " + bestFitness + " | Edges: " + bestDag.getNumEdges());
        newPopulation[0] = bestIndividual.clone();

        // Add the best individual of the last iteration to the new population
        int bestIndex = 1;
        double bestFitness = fitness[1];
        for (int i = 2; i < populationSize; i++) {
            if (fitness[i] < bestFitness) {
                bestFitness = fitness[i];
                bestIndex = i;
            }
        }
        //System.out.println("Best individual of the last iteration: " + bestFitness);
        newPopulation[1] = population[bestIndex];

        tournamentCrossover(newPopulation);
        //uniformCrossover(newPopulation);

        /*for (int i = 0; i < populationSize; i++) {
            double x = (double) treeWidths[i] / maxTreewidth;
            // Number of trues
            int numTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    numTrues++;
                }
            }
            System.out.println(i + " | FIT " + String.format("%.1f", fitness[i]) + " | TW " + treeWidths[i] + " | x " + x + " | Edges: " + numTrues);
        }*/

    }

    /** Uniform crossover with roulette wheel selection */
    private void uniformCrossover(boolean[][] newPopulation) {
        double[] cumulativeProbs = new double[populationSize];
        for (int i = 0; i < populationSize; i++) {
            cumulativeProbs[i] = 1/fitness[i];
        }
        double sum = 0;
        for (int i = 0; i < populationSize; i++) {
            sum += cumulativeProbs[i];
            cumulativeProbs[i] = sum;
        }

        for (int i = 2; i < populationSize; i++) {
            int ind1 = getIndividualByCumulative(cumulativeProbs);
            int ind2 = getIndividualByCumulative(cumulativeProbs);
            for (int j = 0; j < totalEdges; j++) {
                if (random.nextBoolean()) {
                    newPopulation[i][j] = population[ind1][j];
                } else {
                    newPopulation[i][j] = population[ind2][j];
                }
            }
        }
        population = newPopulation;
    }

    private int getIndividualByCumulative(double[] cumulativeProbs) {
        double randomValue = random.nextDouble();
        for (int k = 0; k < populationSize; k++) {
            if (randomValue <= cumulativeProbs[k]) {
                return k;
            }
        }
        return cumulativeProbs.length-1;
    }

    /** Tournament selection and crossover */
    private void tournamentCrossover(boolean[][] newPopulation) {
        for (int i = 2; i < populationSize; i++) {
            int index1 = random.nextInt(populationSize);
            int index2 = random.nextInt(populationSize);
            int index3 = random.nextInt(populationSize);
            int index4 = random.nextInt(populationSize);
            int crossoverPoint = random.nextInt(totalEdges);

            if (fitness[index1] < fitness[index2]) {
                System.arraycopy(population[index1], 0, newPopulation[i], 0, crossoverPoint);
            } else {
                System.arraycopy(population[index2], 0, newPopulation[i], 0, crossoverPoint);
            }
            if (fitness[index3] < fitness[index4]) {
                System.arraycopy(population[index3], crossoverPoint, newPopulation[i], crossoverPoint, totalEdges-crossoverPoint);
            } else {
                System.arraycopy(population[index4], crossoverPoint, newPopulation[i], crossoverPoint, totalEdges-crossoverPoint);
            }
        }
        population = newPopulation;
    }

    private void mutate() {
        for (int i = 0; i < populationSize; i++) {
            double x = (double) treeWidths[i] / maxTreewidth;

            // Number of trues
            int numTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    numTrues++;
                }
            }

            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    double probRemove;
                    if (x < 1) {
                        probRemove = Math.pow(x/10, 2);
                    } else {
                        probRemove = (Math.log10(x)/3) + 0.01;
                    }

                    if (random.nextDouble() < probRemove) {
                        population[i][j] = false;
                    }
                } else {
                    double probAdd;
                    if (x < 1) {
                        probAdd = Math.pow((x-1)/2, 2) + 0.01;
                    } else {
                        probAdd = 0.01/(x+0.01);
                    }

                    if (random.nextDouble() < probAdd) {
                        population[i][j] = true;
                    }
                }
            }

            /* final number of trues
            int finalNumTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    finalNumTrues++;
                }
            }
            System.out.println(i + " | FIT " + String.format("%.1f",fitness[i]) + " | TW " + treeWidths[i] + " | x " + x + " | Edges: " + numTrues + ", " + finalNumTrues); */
        }
    }

    private double calculateFitness(int index, int recursive) {
        Dag union;
        if (candidatesFromInitialDAGs) {
            // Create the DAG that corresponds to each individual
            ArrayList<Dag> candidates = fromChromosomeToDags(population[index]);
            union = applyUnion(alpha, candidates);

            // Check cycles
            if (!union.findCycle().isEmpty()) {
                return Double.MAX_VALUE;
            }
        }
        else {
            // Create the DAG that corresponds to the individual
            union = new Dag(alpha);
            for (int i = 0; i < totalEdges; i++) {
                if (population[index][i]) {
                    union.addEdge(edgesAlpha.get(i));
                }
            }
        }

        // Get the cliques
        treeWidths[index] = Utils.getTreeWidth(union);

        double fitness;
        if (treeWidths[index] > maxTreewidth) {
            //population[index] = fixIndividual(population[index], cliques, recursive);
            //return calculateFitness(index, recursive+1);
            fitness = Utils.SMHD(union, fusionUnion) * ((double)  treeWidths[index] / maxTreewidth);
        } else {
            fitness = Utils.SMHD(union, fusionUnion);
        }

        if (fitness < bestFitness && treeWidths[index] <= maxTreewidth) {
            bestIndividual = population[index].clone();
            bestDag = union;
            bestFitness = fitness;
        }

        return fitness;
    }

    private ArrayList<Dag> fromChromosomeToDags(boolean[] individual) {
        ArrayList<Dag> dags = new ArrayList<>();
        int cumulativeEdges = 0;
        for (int i = 0; i < nDags; i++) {
            Dag dag = new Dag(alpha);
            for (int j = 0; j < nEdges[i]; j++) {
                if (individual[cumulativeEdges]) {
                    dag.addEdge(edgesOriginal.get(cumulativeEdges));
                }
                cumulativeEdges++;
            }
            dags.add(dag);
        }

        return dags;
    }

    private boolean[] fixIndividual(boolean[] individual, Map<Node, Set<Node>> cliques, int recursive) {
        boolean[] fixed = individual.clone();
        for (Node node : cliques.keySet()) {
            Set<Node> clique = cliques.get(node);
            if (clique.size() > maxTreewidth - recursive) {
                int numToRemove = clique.size() - maxTreewidth + recursive;
                for (Node neighbor : clique) {
                    if (numToRemove == 0) {
                        break;
                    }
                    if (neighbor.equals(node)) {
                        continue;
                    }
                    boolean removed = false;

                    Integer position1 = edgePosition.get(new Edge(node, neighbor, Endpoint.TAIL, Endpoint.ARROW));
                    if (position1!=null && fixed[position1]) {
                        fixed[position1] = false;
                        removed = true;
                    }
                    Integer position2 = edgePosition.get(new Edge(neighbor, node, Endpoint.TAIL, Endpoint.ARROW));
                    if (position2!=null && fixed[position2]) {
                        fixed[position2] = false;
                        removed = true;
                    }
                    if (removed) {
                        numToRemove--;
                    }
                    Set<Node> neighborClique = cliques.get(neighbor);
                    if (neighborClique != null) {
                        neighborClique.remove(node);
                    }
                }
            }
        }
        return fixed;
    }
}
