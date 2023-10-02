package org.albacete.simd.bayesfl.data;

import edu.cmu.tetrad.data.DataReader;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DelimiterType;

import java.io.File;
import java.io.IOException;

public class BN_DataSet implements Data {

    private DataSet data;

    private String originalBNPath;

    private final String name;
    
    public BN_DataSet() {
        this.name = "";
    }

    public BN_DataSet(String path, String name) {
        this.data = readData(path);
        this.name = name;
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

    @Override
    public String getName() {
        return name;
    }

    /**
     * Returns the path of the original BN. Used only in experiments for the stats.
     * @return Path of the original BN.
     */
    public String getOriginalBNPath() {
        return this.originalBNPath;
    }

    /**
     * Sets the path of the original BN. Used only in experiments for the stats.
     * @param path Path of the original BN.
     */
    public void setOriginalBNPath(String path) {
        this.originalBNPath = path;
    }
}
