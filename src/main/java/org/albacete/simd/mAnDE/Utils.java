package org.albacete.simd.mAnDE;

import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree2;
import weka.core.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;


/**
 *    LogitBoost Classifier
 */
class LogitBoost2 extends LogitBoost {

    /**
     * Gets all of the classifiers of the ensemble.
     *
     * @return an array with the classifiers of the ensemble.
     */
    public Classifier[] getClassifiers() {
        return m_Classifiers.get(0);
    }

}


/**
 *    AdaBoost Classifier
 */
class AdaBoostM1_2 extends AdaBoostM1 {

    /**
     * Gets all of the classifiers of the ensemble.
     *
     * @return an array with the classifiers of the ensemble.
     */
    public Classifier[] getClassifiers() {
        return m_Classifiers;
    }

}


/**
 *    Bagging Classifier
 */
class Bagging2 extends Bagging {

    /**
     * The size of each bag sample, as a percentage of the training size
     *
     * Changued from int to double
     */
    protected double m_BagSizePercentDouble = 100;

    /**
     * Returns a training set for a particular iteration.
     *
     * Changued to use the variable m_BagSizePercentDouble instead of
     * m_BagSizePercent
     *
     * @param iteration the number of the iteration for the requested training
     * set.
     * @return the training set for the supplied iteration number
     * @throws Exception if something goes wrong when generating a training set.
     */
    @Override
    protected synchronized Instances getTrainingSet(int iteration) {

        Debug.Random r = new Debug.Random(m_Seed + iteration);

        // create the in-bag indicator array if necessary
        if (m_CalcOutOfBag) {
            m_inBag[iteration] = new boolean[m_data.numInstances()];
            return m_data.resampleWithWeights(r, m_inBag[iteration], getRepresentCopiesUsingWeights(), m_BagSizePercentDouble);
        } else {
            return m_data.resampleWithWeights(r, null, getRepresentCopiesUsingWeights(), m_BagSizePercentDouble);
        }
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercentDouble the bag size, as a percentage.
     */
    public void setBagSizePercentDouble(double newBagSizePercentDouble) {

        m_BagSizePercentDouble = newBagSizePercentDouble;
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public double getBagSizePercentDouble() {

        return m_BagSizePercentDouble;
    }

    /**
     * Gets all of the classifiers of the ensemble.
     *
     * @return an array with the classifiers of the ensemble.
     */
    public Classifier[] getClassifiers() {
        return m_Classifiers;
    }
}


/**
 *    RandomForest Classifier
 */
class RandomForest2 extends RandomForest {

    /**
     * The size of each bag sample, as a percentage of the training size
     *
     * Changued from int to double
     */
    protected double m_BagSizePercentDouble = 100;


    /**
     * Constructor that sets base classifier for bagging to RandomTre and default
     * number of iterations to 100.
     */
    public RandomForest2() {
        super();

        RandomTree2 rTree = new RandomTree2();
        rTree.setDoNotCheckCapabilities(true);
        super.m_Classifier = rTree;
    }


    /**
     * Returns a training set for a particular iteration.
     *
     * Changued to use the variable m_BagSizePercentDouble instead of
     * m_BagSizePercent
     *
     * @param iteration the number of the iteration for the requested training
     * set.
     * @return the training set for the supplied iteration number
     */
    @Override
    protected synchronized Instances getTrainingSet(int iteration) {
        Debug.Random r = new Debug.Random(m_Seed + iteration);

        // create the in-bag indicator array if necessary
        if (m_CalcOutOfBag) {
            m_inBag[iteration] = new boolean[m_data.numInstances()];
            return m_data.resampleWithWeights(r, m_inBag[iteration], getRepresentCopiesUsingWeights(), m_BagSizePercentDouble);
        } else {
            return m_data.resampleWithWeights(r, null, getRepresentCopiesUsingWeights(), m_BagSizePercentDouble);
        }
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercentDouble the bag size, as a percentage.
     */
    public void setBagSizePercentDouble(double newBagSizePercentDouble) {
        m_BagSizePercentDouble = newBagSizePercentDouble;
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public double getBagSizePercentDouble() {
        return m_BagSizePercentDouble;
    }

    /**
     * Gets all of the classifiers of the ensemble.
     *
     * @return an array with the classifiers of the ensemble.
     */
    public Classifier[] getClassifiers() {
        return m_Classifiers;
    }


    public void toSP1DE(ConcurrentHashMap<Object, mSPnDE> mSPnDEs) {
        List<Classifier> trees = Arrays.asList(m_Classifiers);

        trees.parallelStream().forEach((tree) -> {
            ((RandomTree2)tree).toSP1DE(mSPnDEs);
        });
    }

    public ArrayList<HashMap<Integer, NodeInt>> toSP2DEgraph() {
        ArrayList<HashMap<Integer, NodeInt>> treeGraph = new ArrayList<>();

        List<Classifier> trees = Arrays.asList(m_Classifiers);

        trees.forEach((tree) -> {
            treeGraph.add(((RandomTree2)tree).toSP2DE());
        });

        return treeGraph;
    }
}
