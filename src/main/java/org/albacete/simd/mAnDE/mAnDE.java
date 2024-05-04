/*
 *  The MIT License (MIT)
 *  
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */
/**
 *    mAnDETree.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */
package org.albacete.simd.mAnDE;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.LMT;
import static weka.classifiers.AbstractClassifier.runClassifier;

import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Option;

public class mAnDE extends AbstractClassifier implements
        OptionHandler, Serializable {

    /**
     * For serialisation.
     */
    private static final long serialVersionUID = 3545430914549890589L;

    // Auxiliary variables //
    /**
     * Instances.
     */
    protected static Instances data;

    /**
     * The discretisation filter.
     */
    protected weka.filters.supervised.attribute.Discretize discretizer = null;

    /**
     * HashMap containing the mSPnDEs.
     */
    public ConcurrentHashMap<Object, mSPnDE> mSPnDEs;

    /**
     * Number of values per variable.
     */
    public static int[] varNumValues;

    /**
     * Number of values of the class.
     */
    public static int classNumValues;

    /**
     * Index of the class.
     */
    public static int y;

    /**
     * Number of instances.
     */
    public static int numInstances;

    /**
     * Naive Bayes for NB mode.
     */
    private static NaiveBayes nb;

    /**
     * Indicates whether Naive Bayes mode is enabled.
     */
    private boolean modeNB = false;

    // mAnDETree parameters //
    /**
     * Configure the base classifier in ensembles
     */
    private String baseClass = "Base";

    /**
     * If desired, configure the ensemble classifier
     */
    private String ensemble = "Bagging";
    
    /**
     * Prune decision trees.
     */
    private boolean pruning = true;

    /**
     * Percentage of instances to be used to make each tree in the assembly.
     */
    private double bagSize = 100;
    
    /**
     * Number of trees that the algorithm make in the structural learning.
     */
    private int nTrees = 100;

    /**
     * n of the mAnDETree.
     */
    private int n = 1;
    
    private double addNB = 0.4;
    
    /**
     * Minimum number of Instances to create a tree.
     */
    private final int minimumInstances = 3;

    /**
     * Create the structure of the classifier taking into account the
     * established parameters.
     *
     * @param instances Instances to classify.
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        checkData(instances);

        buildTrees();

        calculateTables_mSPnDEs();
    }

    /**
     * Calculates the probabilities of class membership for the provided Test
     * Instance.
     *
     * @param instance Instance to classify.
     * @return Probability distribution of predicted class membership.
     * @throws java.lang.Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] res = new double[classNumValues];

        final Instance instance_d;
        discretizer.input(instance);
        instance_d = discretizer.output();

        if (modeNB) {
            return nb.distributionForInstance(instance_d);
        }
        
        // Add up all the probabilities of the mSPnDEs
        mSPnDEs.values().parallelStream().forEach((spode) -> {
            double[] temp = spode.probsForInstance(instance_d);

            for (int i = 0; i < res.length; i++) {
                res[i] += temp[i];
            }
        });

        if (getAddNB() != 0) {
            double percentaje = getAddNB() * mSPnDEs.size();
            double[] temp = nb.distributionForInstance(instance_d);
            for (int i = 0; i < res.length; i++) {
                res[i] += percentaje * temp[i];
            }
        }

        /* Normalize the result. If the sum is 0 (Utils.normalize will return 
         * an IllegalArgumentException), we set the same value in each 
         * possible value of the class.
         */
        try {
            Utils.normalize(res);
        } catch (IllegalArgumentException ex) {
            Arrays.fill(res, 1.0 / classNumValues);
        }
        
        return res;
    }

    public void checkData(Instances instances) throws Exception {
        // Can the classifier work with this data?
        getCapabilities().testWithFail(instances);

        // Delete instances with no class
        instances.deleteWithMissingClass();

        // We driscretise
        discretizer = new weka.filters.supervised.attribute.Discretize();
        discretizer.setInputFormat(instances);
        data = weka.filters.Filter.useFilter(instances, discretizer);
        numInstances = data.numInstances();
    }

    public void buildTrees() throws Exception {
        // We initialize the index dictionaries
        //initializeNameToIndex();

        // We check that with this bagSize, we will have more than
        // minimumInstances instances (default 3)
        if (bagSize > 0) {
            while ((numInstances * (bagSize/100)) < minimumInstances) {
                bagSize = 2 * bagSize;
            }
        }

        // Create the mSPnDE's
        build_mSPnDEs();

        // If we have not created mSPnDE's
        /*if (mSPnDEs.isEmpty()) {
            HashSet<String> done = new HashSet<>();
            done.add(getEnsemble());

            setBagSize(100);
            setBaseClass("J48");
            setAddNB(0.4);

            String[] posClassifiers = {"Bagging", "RF", "AdaBoost"};

            // We try with Bagging, RF and Boosting
            for (int i = 0; i < posClassifiers.length; i++) {
                if (done.contains(posClassifiers[i])) {
                    continue;
                }

                setEnsemble(posClassifiers[i]);
                done.add(posClassifiers[i]);

                // Try to build the mSPnDEs with the new configuration
                //try {
                build_mSPnDEs();
                //} catch (Exception ex) {}

                // If we have created mSPnDEs, we end up with
                if (!mSPnDEs.isEmpty()) {
                    break;
                }
            }

            // If nothing works, we run NB
            if (mSPnDEs.isEmpty()) {
                modeNB = true;
                nb = new NaiveBayes();
                nb.buildClassifier(data);
            }
        }*/
    }

    /**
     * Create the necessary mSPnDE's, by running the trees set in the options...
     */
    private void build_mSPnDEs() throws Exception {
        mSPnDEs = new ConcurrentHashMap<>();

        List<Classifier> trees;
        Classifier base;
        
        switch (baseClass) {
            case "REPTree":
                base = new REPTree();
                if (!pruning) {
                    ((REPTree)base).setNoPruning(true);
                }
                break;
            case "LMT":
                base = new LMT();
                ((LMT)base).setNumBoostingIterations(nTrees);
                break;
            default:
                base = new J48();
                if (!pruning) {
                    ((J48)base).setUnpruned(true);
                }
                break;
        }

        if (!ensemble.equals("none")) {

            switch (getEnsemble()) {
                case "Bagging":
                    Bagging2 bagging = new Bagging2();
                    bagging.setNumExecutionSlots(0);
                    bagging.setClassifier(base);
                    bagging.setNumIterations(nTrees);
                    bagging.setBagSizePercentDouble(bagSize);
                    bagging.buildClassifier(data);
                    trees = Arrays.asList(bagging.getClassifiers());
                    break;
                case "AdaBoost":
                    AdaBoostM1_2 ab = new AdaBoostM1_2();
                    ab.setClassifier(base);
                    ab.setNumIterations(nTrees);
                    ab.buildClassifier(data);
                    trees = Arrays.asList(ab.getClassifiers());
                    break;
                case "RF":
                    RandomForest2 rf = new RandomForest2();
                    rf.setNumExecutionSlots(0);
                    rf.setNumIterations(nTrees);
                    rf.setBagSizePercentDouble(bagSize);
                    rf.buildClassifier(data);

                    if (n == 1) {
                        rf.toSP1DE(mSPnDEs);
                        return;
                    } else {
                        ArrayList<HashMap<Integer, NodeInt>> nodeTrees = rf.toSP2DEgraph();
                        for (HashMap<Integer, NodeInt> nodeTree : nodeTrees) {
                            graphToSP2DE(nodeTree);
                        }
                    }
                    break;
                case "LogitBoost":
                    LogitBoost2 lb = new LogitBoost2();
                    lb.setClassifier(base);
                    lb.setNumIterations(nTrees);
                    lb.buildClassifier(data);
                    trees = Arrays.asList(lb.getClassifiers());
                    break;
                default:
                    throw new Exception("Ensemble type not supported");
            }
            
            /*trees.stream().forEach((tree) -> {
                graphToSPnDE(treeParser(tree));
            });*/
            
        } else {
            base.buildClassifier(data);
            //graphToSPnDE(treeParser(base));
        }
    }

    /**
     * Executes in parallel the 'buildTables()' functions of each mSPnDE, and
     * terminates when all have executed it.
     */
    public void calculateTables_mSPnDEs() {
        if (getAddNB() != 0 || mSPnDEs.isEmpty())
        {
            nb = new NaiveBayes();
            try {
                nb.buildClassifier(data);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (mSPnDEs.isEmpty()) {
            modeNB = true;
        }
        // If we have not run Naive Bayes, we calculate the mAnDETree tables.
        else {
            // Define global variables
            y = data.classIndex();
            classNumValues = data.classAttribute().numValues();
            varNumValues = new int[data.numAttributes()];
            for (int i = 0; i < varNumValues.length; i++) {
                varNumValues[i] = data.attribute(i).numValues();
            }

            //Calls the mSPnDE function that creates the table for each mSPnDE
            mSPnDEs.values().parallelStream().forEach(mSPnDE::buildTables);
        }

        // We free up the discretized data space
        data.delete();

        // Print data of mSPnDEs created
        /*double var = 0;
        double max = 0;
        double min = Double.POSITIVE_INFINITY;
        for (mSPnDE a : mSPnDEs.values()) {
            if (a.getNChildren() > max)
                max = a.getNChildren();
            if (a.getNChildren() < min)
                min = a.getNChildren();
            var += a.getNChildren();
        }
        System.out.println("mSPnDEs," + mSPnDEs.size() + "," + (var/mSPnDEs.size()) + "," + max + "," + min);*/
    }

    /**
     * Converts a HashMap<String,Node> to a representation of mA2DE.
     */
    private void graphToSP2DE(HashMap<Integer, NodeInt> nodes) {
        nodes.values().forEach((node) -> {
            node.getChildren().values().forEach((child) -> {
                toSP2DE(node.getId(), child.getId(), node.getParent().getId(),
                        node.getChildrenArray(child.getId()), child.getChildrenArray());
            });
        });
    }

    /**
     * Create an mSP2DE with the variables 'parent' and 'child' (linked in the
     * tree parser) if it doesn't already exist, and add the variables
     * 'grandparent', 'brothers' and 'grandchildren' as a dependency.
     *
     * @param parent Name of the parent in the mSP2DE.
     * @param child Name of the child in the mSP2DE.
     * @param grandparent Name of the parent of the parent in the mSP2DE.
     * @param brothers Name of the other children of the father in the mSP2DE.
     * @param grandchildren Name of the children of the child in the mSP2DE.
     */
    private void toSP2DE(Integer parent, Integer child, Integer grandparent,
                         ArrayList<Integer> brothers, ArrayList<Integer> grandchildren) {
        long key;
        if (parent < child) key = (((long)parent) << 32) | (child & 0xffffffffL);
        else key = (((long)child) << 32) | (parent & 0xffffffffL);

        if (!mSPnDEs.containsKey(key)) {
            mSPnDEs.put(key, new mSP2DE(parent, child));
        }
        try {
            mSP2DE elem = ((mSP2DE) mSPnDEs.get(key));
            if (!grandparent.equals(parent)) {
                elem.moreChildren(grandparent);
            }
            elem.moreChildren(brothers);
            elem.moreChildren(grandchildren);
        } catch (NullPointerException ex) {}
    }

    /**
     * Returns the number of nSPnDEs and Variables per nSPnDE.
     *
     * @return The number of nSPnDEs and Variables per nSPnDE.
     * @throws java.io.IOException
     */
    public double[] nSPnDEs_variables() throws IOException {
        double[] res = new double[2];
        res[0] = mSPnDEs.size();
        res[1] = 0;
        mSPnDEs.values().forEach((spode) -> {
            res[1] += (spode.getNChildren() / res[0]);
        });

        File f = new File("temp.txt");
        FileWriter file = new FileWriter(f, true);
        PrintWriter pw = new PrintWriter(file, true);
        pw.println(res[0] + "," + res[1]);
        file.close();

        return res;
    }

    /**
     * @param pruning The pruning hyperparameter to set.
     */
    public void setPruning(boolean pruning) {
        this.pruning = pruning;
    }

    /**
     * @param bagSize The bagSize hyperparameter to be set
     */
    public void setBagSize(double bagSize) {
        this.bagSize = bagSize;
    }

    /**
     * @param n The n hyperparameter to be set
     */
    public void setN(int n) {
        if (n > 0 && n <= 3) {
            this.n = n;
        }
    }

    /**
     * @param baseClass The baseClass hyperparameter to set
     */
    public void setBaseClass(String baseClass) {
        this.baseClass = baseClass;
    }
    
    /**
     * @param nTrees The nTrees to be set
     */
    public void setnTrees(int nTrees) {
        this.nTrees = nTrees;
    }
    
    /**
     * @param ensemble The ensemble to set
     */
    public void setEnsemble(String ensemble) {
        this.ensemble = ensemble;
    }
    
    /**
     * @param addNB The addNB to set
     */
    public void setAddNB(double addNB) {
        this.addNB = addNB;
    }

    /**
     * @return The pruning
     */
    public boolean isPruning() {
        return pruning;
    }

    /**
     * @return The bagSize
     */
    public double getBagSize() {
        return bagSize;
    }

    /**
     * @return The n
     */
    public int getN() {
        return n;
    }
    
    /**
     * @return The baseClass
     */
    public String getBaseClass() {
        return baseClass;
    }
    
    /**
     * @return The nTrees
     */
    public int getnTrees() {
        return nTrees;
    }
    
    /**
     * @return The ensemble
     */
    public String getEnsemble() {
        return ensemble;
    }
    
    /**
     * @return The addNB
     */
    public double getAddNB() {
        return addNB;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return The capabilities of this classifier.
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    @Override
    public Enumeration listOptions() {
        Vector newVector = new Vector(6);

        newVector.addElement(new Option("\tn of the mAnDETree (1 or 2, default 1)\n", "N", 1, "-N <int>"));
        newVector.addElement(new Option("\tUse REPTree trees instead of J48 trees", "REP", 0, "-REP"));
        newVector.addElement(new Option("\tNOT performs pruning of decision trees", "P", 0, "-P"));
        newVector.addElement(new Option("\tRealise an ensemble of decision trees", "E", 0, "-E"));
        newVector.addElement(new Option("\tRealise the ensemble of decision trees using Random Forest", "RF", 0, "-RF"));
        newVector.addElement(new Option("\tSet the number of instances used to create each tree when using ensembles (0, 100]\n", "B", 100, "-B <double>"));

        return newVector.elements();
    }

    /**
     * Convert a list of options to the hyperparameters of the algorithm.
     *
     * @param options Options to convert
     * @throws java.lang.Exception
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String N = Utils.getOption('N', options);
        if (N.length() != 0) {
            n = Integer.parseInt(N);
            if (n < 1 || n > 2) {
                n = 1;
            }
        } else {
            n = 1;
        }

        pruning = !Utils.getFlag("P", options);

        String Bag = Utils.getOption('B', options);
        if (Bag.length() != 0) {
            bagSize = Integer.parseInt(Bag);
        } else {
            bagSize = 100;
        }

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Devuelve las opciones actuales del clasificador.
     *
     * @return Un array de strings que se pueda añadir a setOptions
     */
    @Override
    public String[] getOptions() {
        Vector result = new Vector();

        result.add("-N");
        result.add("" + n);

        if (!pruning) {
            result.add("-P");
        }

        result.add("-B");
        result.add("" + bagSize);

        return (String[]) result.toArray(new String[result.size()]);
    }

    /**
     * Main method to test this class.
     *
     * @param args options
     */
    public static void main(String[] args) {
        runClassifier(new mAnDE(), args);
    }

}
