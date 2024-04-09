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
 *    BN_FusionUnion.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.fusion;

import static consensusBN.ConsensusUnion.fusionUnion;
import edu.cmu.tetrad.graph.*;
import bayesfl.model.BN;
import bayesfl.model.Model;

import java.util.ArrayList;

public class BN_FusionUnion implements Fusion {

    private String mode = "Union";

    private String limit = "0";

    @Override
    public Model fusion(Model model1, Model model2) {
        if (!(model1 instanceof BN) || !(model2 instanceof BN)) {
            throw new IllegalArgumentException("The models must be objects of the BN class to use BN_FusionUnion");
        }

        ArrayList<Dag> dags = new ArrayList<>();
        dags.add(((BN) model1).getModel());
        dags.add(((BN) model2).getModel());

        return new BN(fusionUnion(dags, this.mode, this.limit));
    }

    @Override
    public Model fusion(Model [] models) {
        for (Model model : models) {
            if (!(model instanceof BN)) {
                throw new IllegalArgumentException("The models must be objects of the BN class to use BN_FusionUnion");
            }
        }

        ArrayList<Dag> dags = new ArrayList<>();
        for (Model model : models) {
            dags.add(((BN) model).getModel());
        }

        return new BN(fusionUnion(dags, this.mode, this.limit));
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    public void setLimit(String limit) {
        this.limit = limit;
    }
}
