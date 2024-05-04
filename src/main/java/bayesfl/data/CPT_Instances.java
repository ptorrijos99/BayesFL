package bayesfl.data;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Random;

public class CPT_Instances implements Data {

    Instances train;

    Instances test;

    private final String name;

    public CPT_Instances() {
        this.name = "";
    }

    public CPT_Instances(String name) {
        this.name = name;
    }

    public CPT_Instances(Instances train, Instances test, String name) {
        this.train = train;
        this.test = test;
        this.name = name;
    }

    public static ArrayList<Instances> divideDataSet(Instances data, int N, int folds, Random random) {
        data.randomize(random);
        data.stratify(N);

        int actualSampleSize = data.numInstances();
        float bucketSize = actualSampleSize / (float) N;
        ArrayList<Instances> dataSets = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            Instances dataSet = new Instances(data, 0);
            for (int j = 0; j < bucketSize; j++) {
                dataSet.add(data.instance(i * (int) bucketSize + j));
            }
            dataSet.randomize(random);
            dataSet.stratify(folds);

            dataSets.add(dataSet);
        }

        return dataSets;
    }

    public static Instances readData(String bbdd) {
        // Read bbdd
        ConverterUtils.DataSource loader;
        Instances data;
        try {
            loader = new ConverterUtils.DataSource(bbdd);
            data = loader.getDataSet();
            data.setClassIndex(data.numAttributes()-1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return data;
    }

    @Override
    public Object getData() {
        return this.train;
    }

    @Override
    public void setData(Object data) {
        if (!(data instanceof Instances)) {
            throw new IllegalArgumentException("The data must be object of the Instances class");
        }

        System.out.println("Setting data\n\n");

        this.train = (Instances) data;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public int getNInstances() {
        return this.train.numInstances();
    }

    public Instances getTest() {
        return test;
    }

    public void setTest(Instances test) {
        this.test = test;
    }

    public int getNInstancesTest() {
        return this.test.numInstances();
    }

    public int getNAttributes() {
        return this.train.numAttributes();
    }
}
