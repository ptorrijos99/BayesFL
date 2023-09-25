package org.bayesfl.data;

import edu.cmu.tetrad.data.DataReader;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DelimiterType;

import java.io.File;
import java.io.IOException;

public class BN_DataSet implements Data {

    private DataSet data;

    public BN_DataSet(String path) {
        this.data = readData(path);
    }

    private static DataSet readData (String path) {
        // Initial Configuration
        DataReader reader = new DataReader();
        reader.setDelimiter(DelimiterType.COMMA);
        reader.setMaxIntegralDiscrete(100);
        DataSet dataSet = null;

        // Reading data
        try {
            dataSet = reader.parseTabular(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataSet;
    }

    @Override
    public DataSet getData() {
        return data;
    }

    @Override
    public void setData(Object data) {
        if (!(data instanceof DataSet)) {
            throw new IllegalArgumentException("The data must be object of the DataSet class");
        }

        this.data = (DataSet) data;
    }
}
