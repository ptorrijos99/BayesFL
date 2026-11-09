package bayesfl.experiments;

import bayesfl.data.Weka_Instances;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.IntStream;

/**
 * Generative AnDE with the prediction rule of Webb et al. (2012): the joint estimates of all SPnDE components
 * are averaged and normalized once. Counts are Laplace-smoothed; a missing parent value is one more category
 * and a missing child value is skipped, as in Weka's NaiveBayes, so that n = 0 is exactly Weka NB.
 *
 * <p>Summing the client count tables gives the centralized counts, so the federated generative model equals the
 * centralized one for any partition: we evaluate it once per fold (C = 1, kind "fed") and evaluate the local
 * models of every client on the global test fold (kind "local", mean over clients), on the same splits as
 * {@link CCBNExperiment} ({@link Weka_Instances#divide}).</p>
 *
 * <pre>java -cp target/bayesfl-1.0-jar-with-dependencies.jar:classes bayesfl.experiments.WebbAnDEExperiment
 *      Car_Evaluation results/gen_webb.csv [maxN] [folder]</pre>
 */
public class WebbAnDEExperiment {

    static final int[] CLIENTS = {5, 10, 20, 50, 100};

    /** Webb's AnDE of order n, fitted from Laplace-smoothed counts. */
    static final class WebbAnDE {
        final int n, a, o, classIndex;
        final int[] attr;          // attribute indices (class excluded)
        final int[] rChild;        // number of values of each attribute
        final int[] rParent;       // number of values + 1 (missing as a category)
        final List<int[]> sets = new ArrayList<>();
        double[][] logPrior;       // per P: log P(y, x_P) over o * K_P augmented classes
        double[][][] logCond;      // per P, per attribute position: log P(x_j | y, x_P), size o * K_P * r_j

        WebbAnDE(Instances header, int n) {
            this.n = n;
            this.classIndex = header.classIndex();
            this.o = header.numClasses();
            this.a = header.numAttributes() - 1;
            attr = new int[a];
            rChild = new int[a];
            rParent = new int[a];
            for (int k = 0, j = 0; k < header.numAttributes(); k++) {
                if (k == classIndex) continue;
                attr[j] = k;
                rChild[j] = header.attribute(k).numValues();
                rParent[j] = rChild[j] + 1;
                j++;
            }
            combos(new int[n], 0, 0);
        }

        private void combos(int[] cur, int start, int depth) {
            if (depth == n) {
                sets.add(cur.clone());
                return;
            }
            for (int i = start; i < a; i++) {
                cur[depth] = i;
                combos(cur, i + 1, depth + 1);
            }
        }

        private int kP(int[] P) {
            int k = 1;
            for (int p : P) k *= rParent[p];
            return k;
        }

        private int parentKey(int[] P, int[] x) {
            int key = 0;
            for (int p : P) key = key * rParent[p] + (x[p] < 0 ? rChild[p] : x[p]);
            return key;
        }

        int[] encode(Instance inst) {
            int[] x = new int[a];
            for (int j = 0; j < a; j++) x[j] = inst.isMissing(attr[j]) ? -1 : (int) inst.value(attr[j]);
            return x;
        }

        void fit(Instances data) {
            int S = sets.size();
            int N = data.numInstances();
            logPrior = new double[S][];
            logCond = new double[S][][];
            int[][] X = new int[N][];
            int[] Y = new int[N];
            for (int i = 0; i < N; i++) {
                X[i] = encode(data.instance(i));
                Y[i] = (int) data.instance(i).classValue();
            }
            for (int s = 0; s < S; s++) {
                int[] P = sets.get(s);
                int K = kP(P), A = o * K;
                double[] cnt = new double[A];
                double[][] cond = new double[a][];
                for (int j = 0; j < a; j++) cond[j] = new double[A * rChild[j]];
                for (int i = 0; i < N; i++) {
                    int aug = Y[i] * K + parentKey(P, X[i]);
                    cnt[aug]++;
                    for (int j = 0; j < a; j++) if (X[i][j] >= 0) cond[j][aug * rChild[j] + X[i][j]]++;
                }
                double[] lp = new double[A];
                for (int t = 0; t < A; t++) lp[t] = Math.log((cnt[t] + 1.0) / (N + A));
                logPrior[s] = lp;
                logCond[s] = new double[a][];
                for (int j = 0; j < a; j++) {
                    if (contains(P, j)) continue;
                    int r = rChild[j];
                    double[] c = cond[j];
                    double[] l = new double[c.length];
                    for (int t = 0; t < A; t++) {
                        double tot = 0;
                        for (int v = 0; v < r; v++) tot += c[t * r + v];   // records of t with X_j observed
                        for (int v = 0; v < r; v++) l[t * r + v] = Math.log((c[t * r + v] + 1.0) / (tot + r));
                    }
                    logCond[s][j] = l;
                }
            }
        }

        private static boolean contains(int[] P, int j) {
            for (int p : P) if (p == j) return true;
            return false;
        }

        /** Index of the class with the highest averaged joint. */
        int predict(int[] x) {
            int S = sets.size();
            double[] acc = new double[o];
            double[] lj = new double[S * o];
            double max = Double.NEGATIVE_INFINITY;
            for (int s = 0; s < S; s++) {
                int[] P = sets.get(s);
                int K = kP(P), key = parentKey(P, x);
                for (int y = 0; y < o; y++) {
                    int aug = y * K + key;
                    double v = logPrior[s][aug];
                    for (int j = 0; j < a; j++) {
                        double[] l = logCond[s][j];
                        if (l == null || x[j] < 0) continue;
                        v += l[aug * rChild[j] + x[j]];
                    }
                    lj[s * o + y] = v;
                    if (v > max) max = v;
                }
            }
            for (int s = 0; s < S; s++)
                for (int y = 0; y < o; y++) acc[y] += Math.exp(lj[s * o + y] - max);
            int best = 0;
            for (int y = 1; y < o; y++) if (acc[y] > acc[best]) best = y;
            return best;
        }

        double accuracy(Instances test) {
            int ok = 0;
            for (int i = 0; i < test.numInstances(); i++)
                if (predict(encode(test.instance(i))) == (int) test.instance(i).classValue()) ok++;
            return ok / (double) test.numInstances();
        }
    }

    public static void main(String[] args) throws IOException {
        String name = args[0];
        String out = args[1];
        int maxN = args.length > 2 ? Integer.parseInt(args[2]) : 2;
        String folder = args.length > 3 ? args[3] : "Discretas";
        String path = "res/classification/" + folder + "/" + name + ".arff";
        int nFolds = 5;
        try (PrintWriter pw = new PrintWriter(new FileWriter(out, true))) {
            for (int n = 0; n <= maxN; n++) {
                String nName = n == 0 ? "NB" : "A" + n + "DE";
                long t0 = System.currentTimeMillis();
                for (int seed = 1; seed <= 5; seed++) {
                    // Federated = centralized (summed counts)
                    Instances[][][] cen = Weka_Instances.divide(name, path, nFolds, 1, seed);
                    final int nn = n;
                    double[] fed = IntStream.range(0, nFolds).parallel().mapToDouble(f -> {
                        WebbAnDE m = new WebbAnDE(cen[f][0][0], nn);
                        m.fit(cen[f][0][0]);
                        return m.accuracy(cen[f][0][1]);
                    }).toArray();
                    for (int f = 0; f < nFolds; f++)
                        pw.printf(Locale.ROOT, "%s,%s,%d,%d,%d,fed,%.10f%n", name, nName, 1, seed, f, fed[f]);
                    // Local models on each client, tested on the global test fold
                    for (int C : CLIENTS) {
                        if (name.startsWith("House_Votes") && C == 100) continue;
                        Instances[][][] sp = Weka_Instances.divide(name, path, nFolds, C, seed);
                        for (int f = 0; f < nFolds; f++) {
                            final int ff = f;
                            double mean = IntStream.range(0, C).parallel().mapToDouble(c -> {
                                WebbAnDE m = new WebbAnDE(sp[ff][c][0], nn);
                                m.fit(sp[ff][c][0]);
                                return m.accuracy(sp[ff][c][1]);
                            }).average().orElse(Double.NaN);
                            pw.printf(Locale.ROOT, "%s,%s,%d,%d,%d,local,%.10f%n", name, nName, C, seed, f, mean);
                        }
                    }
                    pw.flush();
                }
                System.out.printf("%s %s done in %.1f s%n", name, nName, (System.currentTimeMillis() - t0) / 1000.0);
            }
        }
    }
}
