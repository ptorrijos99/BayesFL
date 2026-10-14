/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2026 Universidad de Castilla-La Mancha, España
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

package bayesfl.algorithms;

import bayesfl.data.Weka_Instances;
import bayesfl.fusion.Classes_Fusion;
import bayesfl.fusion.SPODE_Fusion_Client;
import bayesfl.fusion.SPODE_Fusion_Server;
import bayesfl.model.Classes;
import bayesfl.model.Model;
import bayesfl.model.SPODE_Model;
import bayesfl.model.SpodeStructure;
import optimize.DifferentiableFunction;
import optimize.FunctionValues;
import org.junit.jupiter.api.Test;
import weka.core.Instances;

import java.io.StringReader;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * SPODE_CCBN: analytic gradient of the restricted objective, global tables written
 * from summed counts, and fused models that keep no local copy of the tables.
 */
class SPODE_CCBNTest {

    // 12 rows with a missing value; clients get rows 0-3, 4-7 and 8-11.
    private static final String ARFF =
        "@relation t\n@attribute a {x,y,z}\n@attribute b {p,q}\n@attribute c {r,s}\n@attribute class {0,1,2}\n@data\n"
        + "x,p,r,0\nx,q,s,1\ny,p,r,1\nz,q,?,2\n"
        + "x,p,s,1\nx,p,r,1\ny,q,s,0\nz,p,r,2\n"
        + "x,q,r,0\ny,q,s,1\nx,p,s,0\ny,p,r,2\n";

    private static Instances full() throws Exception {
        Instances d = new Instances(new StringReader(ARFF));
        d.setClassIndex(d.numAttributes() - 1);
        return d;
    }

    private static Weka_Instances client(Instances full, int from) {
        Instances s = new Instances(full, 0);
        for (int i = from; i < from + 4; i++) s.add(full.instance(i));
        return new Weka_Instances("c" + from, s, s);
    }

    private static Classes classes(Weka_Instances... clients) {
        Model[] local = new Model[clients.length];
        for (int i = 0; i < clients.length; i++) {
            local[i] = new Classes_AnDE(new String[]{"-S", "A1DE"}).buildLocalModel(clients[i]);
        }
        return (Classes) new Classes_Fusion().fusion(local);
    }

    /** Central differences against the analytic gradient on random parameters. */
    private static void checkGradient(SPODE_CCBN alg, int s, double[] f, double[] center, int np) {
        DifferentiableFunction fun = alg.objective(s, f, center);
        Random rnd = new Random(7);
        double[] p = new double[np];
        for (int q = 0; q < np; q++) p[q] = rnd.nextGaussian();
        FunctionValues fv = fun.getValues(p);
        double h = 1e-6;
        for (int q = 0; q < np; q++) {
            double old = p[q];
            p[q] = old + h;
            double up = fun.getValues(p).functionValue;
            p[q] = old - h;
            double down = fun.getValues(p).functionValue;
            p[q] = old;
            assertEquals((up - down) / (2 * h), fv.gradient[q], 1e-5, "gradient of parameter " + q);
        }
    }

    @Test
    void analyticGradientMatchesFiniteDifferences() throws Exception {
        Instances data = full();
        Weka_Instances c0 = client(data, 0), c1 = client(data, 4), c2 = client(data, 8);
        Classes cl = classes(c0, c1, c2);

        for (String scheme : new String[]{"dCCBNr", "wCCBNr"}) {
            SPODE_CCBN alg = new SPODE_CCBN(new String[]{"-S", "A1DE", "-P", scheme, "-I", "2", "-LAMBDA", "0.3"},
                    null, cl);
            SPODE_Model model = (SPODE_Model) alg.buildLocalModel(c0);
            SpodeStructure st = alg.getStructure();
            for (int s = 0; s < st.nSpodes(); s++) {
                double[] f = model.weighted ? model.logTables[s] : null;
                double[] center = new double[st.np[s]];
                for (int q = 0; q < center.length; q++) center[q] = 0.1 * q;
                checkGradient(alg, s, f, center, st.np[s]);
            }
        }
    }

    @Test
    void globalTablesAreTheCentralizedEstimates() throws Exception {
        Instances data = full();
        Weka_Instances[] clients = {client(data, 0), client(data, 4), client(data, 8)};
        Classes cl = classes(clients);
        String[] opts = {"-S", "A1DE", "-P", "wCCBNr", "-I", "3", "-GT"};

        SPODE_Fusion_Server server = new SPODE_Fusion_Server(true, true);
        server.begin();
        SPODE_CCBN[] algs = new SPODE_CCBN[clients.length];
        for (int i = 0; i < clients.length; i++) {
            algs[i] = new SPODE_CCBN(opts, null, cl);
            SPODE_Model local = (SPODE_Model) algs[i].buildLocalModel(clients[i]);
            assertNotNull(local.counts, "round 1 releases counts");
            server.accumulate(local);
        }
        SPODE_Model global = (SPODE_Model) server.finish();

        // Tables of a single client holding all the rows
        SPODE_CCBN central = new SPODE_CCBN(opts, null, cl);
        SPODE_Model all = (SPODE_Model) central.buildLocalModel(new Weka_Instances("all", data, data));
        for (int s = 0; s < global.logTables.length; s++) {
            assertArrayEquals(all.logTables[s], global.logTables[s], 1e-12);
        }

        // The client keeps the global tables by reference and no counts; round 2 reuses them
        SPODE_Model fused = (SPODE_Model) new SPODE_Fusion_Client().fusion(null, global);
        assertNull(fused.counts);
        SPODE_Model round2 = (SPODE_Model) algs[0].buildLocalModel(fused, clients[0]);
        assertNull(round2.counts, "counts are released only in round 1");
        for (int s = 0; s < global.logTables.length; s++) {
            assertEquals(global.logTables[s], round2.logTables[s]);
        }
    }

    @Test
    void localTablesAreNotKeptAfterFusion() throws Exception {
        Instances data = full();
        Weka_Instances c0 = client(data, 0), c1 = client(data, 4);
        Classes cl = classes(c0, c1);
        String[] opts = {"-S", "A1DE", "-P", "wCCBNr", "-I", "3"};

        SPODE_Fusion_Server server = new SPODE_Fusion_Server(true, false);
        server.begin();
        SPODE_CCBN a0 = new SPODE_CCBN(opts, null, cl);
        server.accumulate(a0.buildLocalModel(c0));
        server.accumulate(new SPODE_CCBN(opts, null, cl).buildLocalModel(c1));
        SPODE_Model global = (SPODE_Model) server.finish();
        assertNull(global.logTables);

        SPODE_Model fused = (SPODE_Model) new SPODE_Fusion_Client().fusion(null, global);
        assertNull(fused.logTables);
        // Evaluating the fused model recomputes the local tables from the client's data
        double[][] dist = fused.distributions(c0.getTest(), c0.getTrain());
        for (double[] d : dist) assertEquals(1.0, d[0] + d[1] + d[2], 1e-12);
    }
}
