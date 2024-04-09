package bayesfl.data;

import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.DataReader;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DelimiterType;
import edu.cmu.tetrad.data.VerticalDoubleDataBox;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import org.albacete.simd.utils.Problem;

public class BN_DataSet implements Data {

    private DataSet data;

    private String originalBNPath;

    private final String name;
    
    private Problem problem;
    
    public BN_DataSet() {
        this.name = "";
    }
    
    public BN_DataSet(String name) {
        this.name = name;
    }
    
    public BN_DataSet(String path, String name) {
        this.data = readData(path);
        this.problem = new Problem(data);
        this.name = name;
    }
    
    public BN_DataSet(DataSet data, String name) {
        this.data = data;
        this.problem = new Problem(data);
        this.name = name;
    }

    public static DataSet readData(String path) {
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
    
    public static ArrayList<DataSet> divideDataSet(DataSet data, int N) {
        int actualSampleSize = data.getNumRows();
        float bucketSize = actualSampleSize / (float) N;

        int[] cols = new int[data.getNumColumns()];
        for (int i = 0; i < cols.length; i++) {
            cols[i] = i;
        }
        
        ArrayList<DataSet> dataSets = new ArrayList<>();
        
        for (int currentBucket = 0; currentBucket < N; currentBucket++) {
            int first = (int) Math.ceil(currentBucket * bucketSize);
            int last = (int) Math.ceil(currentBucket * bucketSize + bucketSize);
            
            if (last > actualSampleSize) last = actualSampleSize;

            int[] rows = new int[last-first];
            int index = 0;
            for (int i = first; i < last; i++) {
                rows[index] = i;
                index++;
            }

            dataSets.add(new BoxDataSet(new VerticalDoubleDataBox(data.getDoubleData().getSelection(rows, cols).transpose().toArray()), data.getVariables()));
        }
        return dataSets;
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
        this.problem = new Problem((DataSet) data);
    }

    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public int getNInstances() {
        if (data != null) 
            return data.getNumRows();
        return -1;
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

    public Problem getProblem() {
        return this.problem;
    }
}
