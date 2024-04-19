package org.albacete.simd.mAnDE;

import weka.core.ContingencyTables;
import weka.core.Instances;
import weka.core.SpecialFunctions;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class Discretize2Times extends Discretize {

    final int m_NumBins;

    public Discretize2Times(int bins) {
        this.m_NumBins = bins;
    }

    /**
     * Set cutpoints for a single attribute using MDL.
     *
     * @param index the index of the attribute to set cutpoints for
     * @param data the data to work with
     */
    @Override
    protected void calculateCutPointsByMDL(int index, Instances data) {

        // Sort instances
        data.sort(data.attribute(index));

        // Find first instances that's missing
        int firstMissing = data.numInstances();
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).isMissing(index)) {
                firstMissing = i;
                break;
            }
        }
        // SUPERVISED
        m_CutPoints[index] = cutPointsForSubset(data, index, 0, firstMissing);

        // NO SUPERVISED
        if (m_CutPoints[index] == null) {
            calculateCutPointsByEqualFrequencyBinningNS(index);
        }
    }


    // FUNCTIONS OF NOT SUPERVISED PROTECTED IN WEKA.

    /**
     * Set cutpoints for a single attribute.
     *
     * @param index the index of the attribute to set cutpoints for
     */
    protected void calculateCutPointsByEqualFrequencyBinningNS(int index) {

        // Copy data so that it can be sorted
        Instances data = new Instances(getInputFormat());

        // Sort input data
        data.sort(index);

        // Compute weight of instances without missing values
        double sumOfWeights = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).isMissing(index)) {
                break;
            } else {
                sumOfWeights += data.instance(i).weight();
            }
        }
        double freq;
        double[] cutPoints = new double[m_NumBins - 1];

        freq = sumOfWeights / m_NumBins;
        cutPoints = new double[m_NumBins - 1];


        // Compute break points
        double counter = 0, last = 0;
        int cpindex = 0, lastIndex = -1;
        for (int i = 0; i < data.numInstances() - 1; i++) {

            // Stop if value missing
            if (data.instance(i).isMissing(index)) {
                break;
            }
            counter += data.instance(i).weight();
            sumOfWeights -= data.instance(i).weight();

            // Do we have a potential breakpoint?
            if (data.instance(i).value(index) < data.instance(i + 1).value(index)) {

                // Have we passed the ideal size?
                if (counter >= freq) {

                    // Is this break point worse than the last one?
                    if (((freq - last) < (counter - freq)) && (lastIndex != -1)) {
                        cutPoints[cpindex] = (data.instance(lastIndex).value(index) + data
                                .instance(lastIndex + 1).value(index)) / 2;
                        counter -= last;
                        last = counter;
                        lastIndex = i;
                    } else {
                        cutPoints[cpindex] = (data.instance(i).value(index) + data
                                .instance(i + 1).value(index)) / 2;
                        counter = 0;
                        last = 0;
                        lastIndex = -1;
                    }
                    cpindex++;
                    freq = (sumOfWeights + counter) / ((cutPoints.length + 1) - cpindex);
                } else {
                    lastIndex = i;
                    last = counter;
                }
            }
        }

        // Check whether there was another possibility for a cut point
        if ((cpindex < cutPoints.length) && (lastIndex != -1)) {
            cutPoints[cpindex] = (data.instance(lastIndex).value(index) + data
                    .instance(lastIndex + 1).value(index)) / 2;
            cpindex++;
        }

        // Did we find any cutpoints?
        if (cpindex == 0) {
            m_CutPoints[index] = null;
        } else {
            double[] cp = new double[cpindex];
            for (int i = 0; i < cpindex; i++) {
                cp[i] = cutPoints[i];
            }
            m_CutPoints[index] = cp;
        }
    }


    // ALL OF THIS FUNCTIONS ARE PRIVATE IN WEKA.
    // WE NEED TO COPY THE CODE TO OVERRIDE calculateCutPointsByMDL(int index, Instances data);

    /**
     * Selects cutpoints for sorted subset.
     *
     * @param instances
     * @param attIndex
     * @param first
     * @param lastPlusOne
     * @return
     */
    private double[] cutPointsForSubset(Instances instances, int attIndex,
                                        int first, int lastPlusOne) {

        double[][] counts, bestCounts;
        double[] priorCounts, left, right, cutPoints;
        double currentCutPoint = -Double.MAX_VALUE, bestCutPoint = -1, currentEntropy, bestEntropy, priorEntropy, gain;
        int bestIndex = -1, numCutPoints = 0;
        double numInstances = 0;

        // Compute number of instances in set
        if ((lastPlusOne - first) < 2) {
            return null;
        }

        // Compute class counts.
        counts = new double[2][instances.numClasses()];
        for (int i = first; i < lastPlusOne; i++) {
            numInstances += instances.instance(i).weight();
            counts[1][(int) instances.instance(i).classValue()] += instances
                    .instance(i).weight();
        }

        // Save prior counts
        priorCounts = new double[instances.numClasses()];
        System.arraycopy(counts[1], 0, priorCounts, 0, instances.numClasses());

        // Entropy of the full set
        priorEntropy = ContingencyTables.entropy(priorCounts);
        bestEntropy = priorEntropy;

        // Find best entropy.
        bestCounts = new double[2][instances.numClasses()];
        for (int i = first; i < (lastPlusOne - 1); i++) {
            counts[0][(int) instances.instance(i).classValue()] += instances
                    .instance(i).weight();
            counts[1][(int) instances.instance(i).classValue()] -= instances
                    .instance(i).weight();
            if (instances.instance(i).value(attIndex) < instances.instance(i + 1)
                    .value(attIndex)) {
                currentCutPoint = (instances.instance(i).value(attIndex) + instances
                        .instance(i + 1).value(attIndex)) / 2.0;
                currentEntropy = ContingencyTables.entropyConditionedOnRows(counts);
                if (currentEntropy < bestEntropy) {
                    bestCutPoint = currentCutPoint;
                    bestEntropy = currentEntropy;
                    bestIndex = i;
                    System.arraycopy(counts[0], 0, bestCounts[0], 0,
                            instances.numClasses());
                    System.arraycopy(counts[1], 0, bestCounts[1], 0,
                            instances.numClasses());
                }
                numCutPoints++;
            }
        }

        // Use worse encoding?
        if (!m_UseBetterEncoding) {
            numCutPoints = (lastPlusOne - first) - 1;
        }

        // Checks if gain is zero
        gain = priorEntropy - bestEntropy;
        if (gain <= 0) {
            return null;
        }

        // Check if split is to be accepted
        if ((m_UseKononenko && KononenkosMDL(priorCounts, bestCounts, numInstances,
                numCutPoints))
                || (!m_UseKononenko && FayyadAndIranisMDL(priorCounts, bestCounts,
                numInstances, numCutPoints))) {

            // Select split points for the left and right subsets
            left = cutPointsForSubset(instances, attIndex, first, bestIndex + 1);
            right = cutPointsForSubset(instances, attIndex, bestIndex + 1,
                    lastPlusOne);

            // Merge cutpoints and return them
            if ((left == null) && (right) == null) {
                cutPoints = new double[1];
                cutPoints[0] = bestCutPoint;
            } else if (right == null) {
                cutPoints = new double[left.length + 1];
                System.arraycopy(left, 0, cutPoints, 0, left.length);
                cutPoints[left.length] = bestCutPoint;
            } else if (left == null) {
                cutPoints = new double[1 + right.length];
                cutPoints[0] = bestCutPoint;
                System.arraycopy(right, 0, cutPoints, 1, right.length);
            } else {
                cutPoints = new double[left.length + right.length + 1];
                System.arraycopy(left, 0, cutPoints, 0, left.length);
                cutPoints[left.length] = bestCutPoint;
                System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
            }

            return cutPoints;
        } else {
            return null;
        }
    }

    /**
     * Test using Kononenko's MDL criterion.
     *
     * @param priorCounts
     * @param bestCounts
     * @param numInstances
     * @param numCutPoints
     * @return true if the split is acceptable
     */
    private boolean KononenkosMDL(double[] priorCounts, double[][] bestCounts,
                                  double numInstances, int numCutPoints) {

        double distPrior, instPrior, distAfter = 0, sum, instAfter = 0;
        double before, after;
        int numClassesTotal;

        // Number of classes occuring in the set
        numClassesTotal = 0;
        for (double priorCount : priorCounts) {
            if (priorCount > 0) {
                numClassesTotal++;
            }
        }

        // Encode distribution prior to split
        distPrior = SpecialFunctions.log2Binomial(numInstances + numClassesTotal
                - 1, numClassesTotal - 1);

        // Encode instances prior to split.
        instPrior = SpecialFunctions.log2Multinomial(numInstances, priorCounts);

        before = instPrior + distPrior;

        // Encode distributions and instances after split.
        for (double[] bestCount : bestCounts) {
            sum = Utils.sum(bestCount);
            distAfter += SpecialFunctions.log2Binomial(sum + numClassesTotal - 1,
                    numClassesTotal - 1);
            instAfter += SpecialFunctions.log2Multinomial(sum, bestCount);
        }

        // Coding cost after split
        after = Utils.log2(numCutPoints) + distAfter + instAfter;

        // Check if split is to be accepted
        return (before > after);
    }

    /**
     * Test using Fayyad and Irani's MDL criterion.
     *
     * @param priorCounts
     * @param bestCounts
     * @param numInstances
     * @param numCutPoints
     * @return true if the splits is acceptable
     */
    private boolean FayyadAndIranisMDL(double[] priorCounts,
                                       double[][] bestCounts, double numInstances, int numCutPoints) {

        double priorEntropy, entropy, gain;
        double entropyLeft, entropyRight, delta;
        int numClassesTotal, numClassesRight, numClassesLeft;

        // Compute entropy before split.
        priorEntropy = ContingencyTables.entropy(priorCounts);

        // Compute entropy after split.
        entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

        // Compute information gain.
        gain = priorEntropy - entropy;

        // Number of classes occuring in the set
        numClassesTotal = 0;
        for (double priorCount : priorCounts) {
            if (priorCount > 0) {
                numClassesTotal++;
            }
        }

        // Number of classes occuring in the left subset
        numClassesLeft = 0;
        for (int i = 0; i < bestCounts[0].length; i++) {
            if (bestCounts[0][i] > 0) {
                numClassesLeft++;
            }
        }

        // Number of classes occuring in the right subset
        numClassesRight = 0;
        for (int i = 0; i < bestCounts[1].length; i++) {
            if (bestCounts[1][i] > 0) {
                numClassesRight++;
            }
        }

        // Entropy of the left and the right subsets
        entropyLeft = ContingencyTables.entropy(bestCounts[0]);
        entropyRight = ContingencyTables.entropy(bestCounts[1]);

        // Compute terms for MDL formula
        delta = Utils.log2(Math.pow(3, numClassesTotal) - 2)
                - ((numClassesTotal * priorEntropy) - (numClassesRight * entropyRight) - (numClassesLeft * entropyLeft));

        // Check if split is to be accepted
        return (gain > (Utils.log2(numCutPoints) + delta) / numInstances);
    }

}
