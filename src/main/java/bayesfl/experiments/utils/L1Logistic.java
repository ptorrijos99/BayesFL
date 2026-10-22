package bayesfl.experiments.utils;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * L1-regularised logistic regression (LIBLINEAR, one-vs-rest) as a WEKA classifier.
 * Nominal attributes become indicators and every input is standardised on the training data.
 */
public class L1Logistic extends AbstractClassifier {

    private final double c;
    private MultiFilter filter;
    private Model model;

    public L1Logistic(double c) {
        this.c = c;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        filter = new MultiFilter();
        filter.setFilters(new Filter[] {new ReplaceMissingValues(), new NominalToBinary(), new Standardize()});
        filter.setInputFormat(data);
        Instances train = Filter.useFilter(data, filter);

        Problem problem = new Problem();
        problem.l = train.numInstances();
        problem.n = train.numAttributes();       // Inputs plus the bias term
        problem.bias = 1;
        problem.x = new Feature[problem.l][];
        problem.y = new double[problem.l];
        for (int i = 0; i < problem.l; i++) {
            problem.x[i] = features(train.instance(i));
            problem.y[i] = train.instance(i).classValue();
        }
        Linear.disableDebugOutput();
        model = Linear.train(problem, new Parameter(SolverType.L1R_LR, c, 0.01));
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        filter.input(instance);
        Instance x = filter.output();
        double[] estimates = new double[model.getNrClass()];
        Linear.predictProbability(model, features(x), estimates);
        double[] dist = new double[instance.numClasses()];
        int[] labels = model.getLabels();
        for (int k = 0; k < labels.length; k++) dist[labels[k]] = estimates[k];
        return dist;
    }

    /** Non-zero inputs with 1-based indices, followed by the bias feature. */
    private static Feature[] features(Instance x) {
        int classIndex = x.classIndex();
        java.util.List<Feature> nodes = new java.util.ArrayList<>();
        int index = 0;
        for (int a = 0; a < x.numAttributes(); a++) {
            if (a == classIndex) continue;
            index++;
            double v = x.value(a);
            if (v != 0) nodes.add(new FeatureNode(index, v));
        }
        nodes.add(new FeatureNode(x.numAttributes(), 1));
        return nodes.toArray(new Feature[0]);
    }
}
