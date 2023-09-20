package org.bayesfl.data;

import edu.cmu.tetrad.data.DataSet;

public class BN_DataSet implements Data {

    private DataSet data;

    @Override
    public DataSet getData() {
        return data;
    }

    @Override
    public void setData(Object data) {
        if (!(data instanceof DataSet)) {
            throw new IllegalArgumentException("The models must be objects of the BN class to use BN_FusionUnion");
        }

        this.data = (DataSet) data;

    }
}
