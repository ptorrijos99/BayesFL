package consensusBN;

import consensusBN.Method.*;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.graph.Dag;
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
    public Boolean repeatCandidates = false;
    public Boolean useSuperGreedy = false;
    public Boolean addEmptySuperGreedy = false;
    public Boolean useMinCutBES = false;
    public boolean useGreedyWarmstart = true;

    /** Pre-run MinCutTreeWidthUnion to reuse as warmstart (avoids redundant computation). */
    public MinCutTreeWidthUnion precomputedMinCut = null;
    /** Index into precomputedMinCut.outputExperiment* for the best snapshot to use. */
    public int precomputedMinCutBestIdx = -1;

    /** Pre-computed per-client DAGs for this TW (injected from ExperimentsJournal pre-run). */
    public List<Dag> cachedGreedyDags = null;
    /** Pre-computed per-client DAGs for TW-1 (avoids second greedy call in initializePopulation). */
    public List<Dag> cachedGreedyDagsM1 = null;
    /** Wall-clock time of the pre-computation (seconds). */
    public double cachedGreedyTime = -1;

    // Best values and stats
    private boolean[] bestIndividual;
    public Dag bestDag;
    public double bestFitness = Double.MAX_VALUE;
    public double executionTime;
    public double executionTimeUnion;
    public double executionTimeGreedy;

    /** Best fitness recorded after each iteration, for convergence analysis. */
    public List<Double> convergenceCurve = new ArrayList<>();

    // "Final" variables
    private final Random random;
    private final List<Node> alpha;
    private final List<Dag> alphaDags;
    private final List<Dag> originalDags;
    private List<Graph> originalMoralGraphs;
    public Dag greedyDag;
    public Dag fusionUnion;
    public Graph moralGraphUnion;

    private double[] fitness;
    private int[] treeWidths;

    // Variables
    public Population method;
    private boolean[][] population;
    private int totalEdges;

    public GeneticTreeWidthUnion(List<Dag> dags, int seed) {
        this.random = new Random(seed);
        this.originalDags = dags;

        // Transform the DAGs to the same alpha order
        alpha = alphaOrder(dags);
        alphaDags = new ArrayList<>();
        for (Dag dag : dags) {
            alphaDags.add(transformToAlpha(dag, alpha));
        }

        // Apply the union of the DAGs
        double startTime = System.currentTimeMillis();
        fusionUnion = applyUnion(alpha, alphaDags);
        moralGraphUnion = Utils.moralize(fusionUnion);
        executionTimeUnion = (System.currentTimeMillis() - startTime) / 1000;
    }

    public GeneticTreeWidthUnion(List<Dag> dags, int seed, int maxTreewidth) {
        this(dags, seed);
        this.maxTreewidth = maxTreewidth;
    }

    public List<Node> getAlpha() { return alpha; }

    /**
     * Complete union of the DAGs limiting the tree-width of the fused DAG.
     * @return The union of the DAGs.
     */
    public Dag fusionUnion() {
        // Reset per-call state so repeated calls on the same instance are independent
        bestFitness = Double.MAX_VALUE;
        bestDag = null;
        convergenceCurve = new ArrayList<>();

        double startTime = System.currentTimeMillis();
        treeWidths = new int[populationSize];

        if (metricAgainstOriginalDAGs && metricSMHD) {
            originalMoralGraphs = new ArrayList<>();
            for (Dag d : originalDags) {
                originalMoralGraphs.add(Utils.moralize(d));
            }
        }

        if (candidatesFromInitialDAGs) {
            if (!repeatCandidates) {
                InitialDAGsWoRepeat_Method idag = new InitialDAGsWoRepeat_Method();
                if (cachedGreedyDags != null) {
                    idag.cachedGreedyDags   = cachedGreedyDags;
                    idag.cachedGreedyDagsM1 = cachedGreedyDagsM1;
                    idag.cachedGreedyTime   = cachedGreedyTime;
                }
                method = idag;
            }
            else {
                if (useMinCutBES) {
                    InitialDAGs_Method idag = new InitialDAGs_Method(true);
                    if (precomputedMinCut != null && precomputedMinCutBestIdx >= 0) {
                        idag.precomputedMinCut = precomputedMinCut;
                        idag.precomputedBestIdx = precomputedMinCutBestIdx;
                    }
                    method = idag;
                } else {
                    InitialDAGs_Method idag = new InitialDAGs_Method();
                    if (cachedGreedyDags != null) {
                        idag.cachedGreedyDags   = cachedGreedyDags;
                        idag.cachedGreedyDagsM1 = cachedGreedyDagsM1;
                        idag.cachedGreedyTime   = cachedGreedyTime;
                    }
                    method = idag;
                }
            }
        }
        else {
            method = new Fusion_Method(useSuperGreedy, addEmptySuperGreedy);
        }

        method.setUseGreedyWarmstart(useGreedyWarmstart);
        method.initialize(originalDags, alpha, alphaDags, maxTreewidth, random);
        population = method.initializePopulation(populationSize);
        totalEdges = population[0].length;
        bestIndividual = population[0].clone();

        for (int i = 0; i < numIterations; i++) {
            evaluate();
            crossover();
            evaluate();
            mutate();
            convergenceCurve.add(bestFitness);
        }

        executionTime = (System.currentTimeMillis() - startTime) / 1000;

        greedyDag = method.getGreedyDag();
        executionTimeGreedy = method.getExecutionTimeGreedy();

        // Fallback: if no individual satisfied the TW constraint.
        // When useGreedyWarmstart=true, use the greedy solution (natural fallback).
        // When useGreedyWarmstart=false (ablation), use an empty DAG: it trivially
        // satisfies any TW constraint, is identical for all algorithms, and honestly
        // reflects that without warmstart the GA found no valid solution.
        if (bestDag == null) {
            if (useGreedyWarmstart) {
                bestDag = greedyDag;
            } else {
                bestDag = new Dag(alpha);
            }
        }

        return bestDag;
    }



    private void evaluate() {
        fitness = new double[populationSize];

        IntStream.range(0, populationSize)
                .parallel()
                .forEach(i -> fitness[i] = calculateFitness(i));

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

    private double calculateFitness(int index) {
        Dag individualDag = method.getUnionFromChromosome(population[index]);
        if (individualDag == null) {
            return Double.MAX_VALUE;
        }

        // Get the cliques
        treeWidths[index] = Utils.getTreeWidth(individualDag);

        double fitness;
        if (metricAgainstOriginalDAGs) {
            if (metricSMHD) {
                Graph moralIndividualGraph = Utils.moralize(individualDag);
                fitness = Utils.SMHDwithoutMoralize(moralIndividualGraph, originalMoralGraphs);
            } else {
                fitness = Utils.fusionSimilarity(individualDag, originalDags);
            }
        } else {
            Graph moralIndividualGraph = Utils.moralize(individualDag);
            fitness = Utils.SMHDwithoutMoralize(moralIndividualGraph, moralGraphUnion);
        }

        if (treeWidths[index] > maxTreewidth) {
            fitness = fitness * ((double)  treeWidths[index] / maxTreewidth);
        }

        if (fitness < bestFitness && treeWidths[index] <= maxTreewidth) {
            bestIndividual = population[index].clone();
            bestDag = individualDag;
            bestFitness = fitness;
        }

        return fitness;
    }
}
