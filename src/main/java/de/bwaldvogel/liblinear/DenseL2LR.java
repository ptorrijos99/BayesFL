package de.bwaldvogel.liblinear;

import java.util.Arrays;

/**
 * L2-regularised logistic regression (liblinear's {@code L2R_LR}) for problems whose rows are
 * dense: row {@code x[i]} holds the values of features {@code 1..n} (the bias column included as
 * the last one, as in a {@link Problem} with {@code bias >= 0} built by hand).
 *
 * <p>It reproduces {@link Linear#train} and {@link Linear#crossValidation} of liblinear 2.44
 * operation by operation: the same {@link Newton} solver, the same order of every floating point
 * sum and product and the same use of {@link Parameter#random}, so the weights (and the
 * cross-validated predictions) are bitwise identical to the sparse ones. Only the storage changes:
 * a {@code double[]} per row instead of a {@link FeatureNode} object per value, which takes four
 * times less memory, and the dot products of four rows are evaluated together, which hides the
 * latency of the floating point additions without changing their order inside each row.
 */
public final class DenseL2LR {

    private DenseL2LR() {
    }

    /** {@link Linear#train} for {@code L2R_LR} on dense rows. */
    public static Model train(double[][] x, double[] y, double bias, Parameter param) {
        if (param.solverType != SolverType.L2R_LR) {
            throw new IllegalArgumentException("only L2R_LR is supported");
        }
        if (param.eps <= 0) throw new IllegalArgumentException("eps <= 0");
        if (param.C <= 0) throw new IllegalArgumentException("C <= 0");
        int l = x.length;
        if (l == 0) throw new IllegalArgumentException("problem has zero instances");
        int n = x[0].length;
        if (n == 0) throw new IllegalArgumentException("problem has zero features");
        if (!param.regularize_bias && bias != 1.0) {
            throw new IllegalArgumentException("To not regularize bias, must specify -B 1 along with -R");
        }

        int w_size = n;
        Model model = new Model();
        model.nr_feature = bias >= 0 ? n - 1 : n;
        model.solverType = param.solverType;
        model.bias = bias;

        int[] perm = new int[l];
        int[][] groups = groupClasses(y, perm);
        int[] label = groups[0];
        int[] start = groups[1];
        int[] count = groups[2];
        int nr_class = label.length;

        model.nr_class = nr_class;
        model.label = Arrays.copyOf(label, nr_class);

        double[] weighted_C = new double[nr_class];
        Arrays.fill(weighted_C, param.C);
        for (int i = 0; i < param.getNumWeights(); i++) {
            int j;
            for (j = 0; j < nr_class; j++)
                if (param.weightLabel[i] == label[j]) break;
            if (j == nr_class) {
                throw new IllegalArgumentException("class label " + param.weightLabel[i] + " specified in weight is not found");
            }
            weighted_C[j] *= param.weight[i];
        }

        double[][] sx = new double[l][];
        for (int i = 0; i < l; i++) sx[i] = x[perm[i]];
        double[] sy = new double[l];

        if (nr_class == 2) {
            model.w = new double[w_size];
            int e0 = start[0] + count[0];
            int k = 0;
            for (; k < e0; k++) sy[k] = +1;
            for (; k < l; k++) sy[k] = -1;
            if (param.init_sol != null) {
                for (int i = 0; i < w_size; i++) model.w[i] = param.init_sol[i];
            }
            trainOne(sx, sy, param, model.w, weighted_C[0], weighted_C[1]);
        } else {
            model.w = new double[w_size * nr_class];
            double[] w = new double[w_size];
            for (int i = 0; i < nr_class; i++) {
                int si = start[i];
                int ei = si + count[i];
                int k = 0;
                for (; k < si; k++) sy[k] = -1;
                for (; k < ei; k++) sy[k] = +1;
                for (; k < l; k++) sy[k] = -1;
                if (param.init_sol != null) {
                    for (int j = 0; j < w_size; j++) w[j] = param.init_sol[j * nr_class + i];
                } else {
                    Arrays.fill(w, 0);
                }
                trainOne(sx, sy, param, w, weighted_C[i], param.C);
                for (int j = 0; j < n; j++) model.w[j * nr_class + i] = w[j];
            }
        }
        return model;
    }

    /** {@link Linear#crossValidation} for {@code L2R_LR} on dense rows. */
    public static void crossValidation(double[][] x, double[] y, double bias, Parameter param, int nr_fold, double[] target) {
        int l = x.length;
        int[] perm = new int[l];
        if (nr_fold > l) nr_fold = l;
        int[] fold_start = new int[nr_fold + 1];
        for (int i = 0; i < l; i++) perm[i] = i;
        for (int i = 0; i < l; i++) {
            int j = i + param.random.nextInt(l - i);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        for (int i = 0; i <= nr_fold; i++) fold_start[i] = i * l / nr_fold;

        for (int i = 0; i < nr_fold; i++) {
            int begin = fold_start[i];
            int end = fold_start[i + 1];
            int subl = l - (end - begin);
            double[][] subx = new double[subl][];
            double[] suby = new double[subl];
            int k = 0;
            for (int j = 0; j < begin; j++, k++) {
                subx[k] = x[perm[j]];
                suby[k] = y[perm[j]];
            }
            for (int j = end; j < l; j++, k++) {
                subx[k] = x[perm[j]];
                suby[k] = y[perm[j]];
            }
            Model submodel = train(subx, suby, bias, param);
            double[] dec = new double[submodel.nr_class];
            for (int j = begin; j < end; j++) target[perm[j]] = predictValues(submodel, x[perm[j]], dec);
        }
    }

    /** {@link Linear#predictValues} on a dense row. */
    public static double predictValues(Model model, double[] x, double[] dec_values) {
        int n = model.bias >= 0 ? model.nr_feature + 1 : model.nr_feature;
        double[] w = model.w;
        int nr_w = model.nr_class == 2 ? 1 : model.nr_class;
        for (int i = 0; i < nr_w; i++) dec_values[i] = 0;
        int len = Math.min(x.length, n);
        for (int j = 0; j < len; j++) {
            double v = x[j];
            for (int i = 0; i < nr_w; i++) dec_values[i] += w[j * nr_w + i] * v;
        }
        if (model.nr_class == 2) {
            return (dec_values[0] > 0) ? model.label[0] : model.label[1];
        }
        int dec_max_idx = 0;
        for (int i = 1; i < model.nr_class; i++) {
            if (dec_values[i] > dec_values[dec_max_idx]) dec_max_idx = i;
        }
        return model.label[dec_max_idx];
    }

    /** The rows of {@code x} as liblinear feature nodes (indices 1..n). */
    public static Feature[] toFeatures(double[] x) {
        Feature[] f = new Feature[x.length];
        for (int j = 0; j < x.length; j++) f[j] = new FeatureNode(j + 1, x[j]);
        return f;
    }

    private static void trainOne(double[][] x, double[] y, Parameter param, double[] w, double Cp, double Cn) {
        int l = x.length;
        double[] C = new double[l];
        int pos = 0;
        for (int i = 0; i < l; i++) {
            if (y[i] > 0) {
                pos++;
                C[i] = Cp;
            } else {
                C[i] = Cn;
            }
        }
        int neg = l - pos;
        double primal_solver_tol = param.eps * Math.max(Math.min(pos, neg), 1) / l;
        DenseLrFunction fun_obj = new DenseLrFunction(x, y, C, param.regularize_bias);
        new Newton(fun_obj, primal_solver_tol, param.max_iters).newton(w);
    }

    /** Linear's private groupClasses: {label, start, count}, filling {@code perm}. */
    private static int[][] groupClasses(double[] y, int[] perm) {
        int l = y.length;
        int max_nr_class = 16;
        int nr_class = 0;
        int[] label = new int[max_nr_class];
        int[] count = new int[max_nr_class];
        int[] data_label = new int[l];
        for (int i = 0; i < l; i++) {
            int this_label = (int) y[i];
            int j;
            for (j = 0; j < nr_class; j++) {
                if (this_label == label[j]) {
                    ++count[j];
                    break;
                }
            }
            data_label[i] = j;
            if (j == nr_class) {
                if (nr_class == max_nr_class) {
                    max_nr_class *= 2;
                    label = Arrays.copyOf(label, max_nr_class);
                    count = Arrays.copyOf(count, max_nr_class);
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }
        if (nr_class == 2 && label[0] == -1 && label[1] == 1) {
            int t = label[0]; label[0] = label[1]; label[1] = t;
            t = count[0]; count[0] = count[1]; count[1] = t;
            for (int i = 0; i < l; i++) data_label[i] = data_label[i] == 0 ? 1 : 0;
        }
        int[] start = new int[nr_class];
        start[0] = 0;
        for (int i = 1; i < nr_class; i++) start[i] = start[i - 1] + count[i - 1];
        for (int i = 0; i < l; i++) {
            perm[start[data_label[i]]] = i;
            ++start[data_label[i]];
        }
        start[0] = 0;
        for (int i = 1; i < nr_class; i++) start[i] = start[i - 1] + count[i - 1];
        return new int[][] {Arrays.copyOf(label, nr_class), start, Arrays.copyOf(count, nr_class)};
    }

    /** L2R_LrFunction (with L2R_ErmFunction) on dense rows. */
    static final class DenseLrFunction implements Function {
        private final double[][] x;
        private final double[] y;
        private final double[] C;
        private final double[] wx;
        private final double[] tmp;
        private final double[] D;
        private final double[] hvCoef;
        private final boolean regularize_bias;
        private final int l;
        private final int n;
        private double wTw;

        DenseLrFunction(double[][] x, double[] y, double[] C, boolean regularize_bias) {
            this.x = x;
            this.y = y;
            this.C = C;
            this.regularize_bias = regularize_bias;
            this.l = x.length;
            this.n = x[0].length;
            wx = new double[l];
            tmp = new double[l];
            D = new double[l];
            hvCoef = new double[l];
        }

        private double C_times_loss(int i, double wx_i) {
            double ywx_i = wx_i * y[i];
            if (ywx_i >= 0)
                return C[i] * Math.log(1 + Math.exp(-ywx_i));
            else
                return C[i] * (-ywx_i + Math.log(1 + Math.exp(ywx_i)));
        }

        /** out[i] = sum_j v[j] * x[i][j], summed in j order. */
        private void Xv(double[] v, double[] out) {
            int i = 0;
            for (; i + 7 < l; i += 8) dot8(v, i, out);
            for (; i < l; i++) out[i] = dot(v, x[i]);
        }

        /** The dot products of rows i..i+7 with v, as eight independent chains of additions. */
        private void dot8(double[] v, int i, double[] out) {
            double[] a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
            double[] e = x[i + 4], f = x[i + 5], g = x[i + 6], h = x[i + 7];
            double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
            for (int j = 0; j < n; j++) {
                double vj = v[j];
                r0 += vj * a[j];
                r1 += vj * b[j];
                r2 += vj * c[j];
                r3 += vj * d[j];
                r4 += vj * e[j];
                r5 += vj * f[j];
                r6 += vj * g[j];
                r7 += vj * h[j];
            }
            out[i] = r0;
            out[i + 1] = r1;
            out[i + 2] = r2;
            out[i + 3] = r3;
            out[i + 4] = r4;
            out[i + 5] = r5;
            out[i + 6] = r6;
            out[i + 7] = r7;
        }

        private double dot(double[] v, double[] a) {
            double r = 0;
            for (int j = 0; j < n; j++) r += v[j] * a[j];
            return r;
        }

        /** out[j] = sum_i coef[i] * x[i][j], accumulated in i order. */
        private void XTv(double[] coef, double[] out) {
            Arrays.fill(out, 0, n, 0);
            int i = 0;
            for (; i + 7 < l; i += 8) axpy8(coef, i, out);
            for (; i < l; i++) axpy1(coef[i], x[i], out);
        }

        /** out[j] += coef[k] * x[k][j] for the rows k = i..i+7, in that order. */
        private void axpy8(double[] coef, int i, double[] out) {
            double[] a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
            double[] e = x[i + 4], f = x[i + 5], g = x[i + 6], h = x[i + 7];
            double c0 = coef[i], c1 = coef[i + 1], c2 = coef[i + 2], c3 = coef[i + 3];
            double c4 = coef[i + 4], c5 = coef[i + 5], c6 = coef[i + 6], c7 = coef[i + 7];
            for (int j = 0; j < n; j++) {
                double t = out[j];
                t += c0 * a[j];
                t += c1 * b[j];
                t += c2 * c[j];
                t += c3 * d[j];
                t += c4 * e[j];
                t += c5 * f[j];
                t += c6 * g[j];
                t += c7 * h[j];
                out[j] = t;
            }
        }

        private void axpy1(double c0, double[] a, double[] out) {
            for (int j = 0; j < n; j++) out[j] += c0 * a[j];
        }

        @Override
        public double fun(double[] w) {
            double f = 0;
            int w_size = n;
            wTw = 0;
            Xv(w, wx);
            for (int i = 0; i < w_size; i++) wTw += w[i] * w[i];
            if (!regularize_bias) wTw -= w[w_size - 1] * w[w_size - 1];
            for (int i = 0; i < l; i++) f += C_times_loss(i, wx[i]);
            f = f + 0.5 * wTw;
            return f;
        }

        @Override
        public void grad(double[] w, double[] g) {
            int w_size = n;
            for (int i = 0; i < l; i++) {
                tmp[i] = 1 / (1 + Math.exp(-y[i] * wx[i]));
                D[i] = tmp[i] * (1 - tmp[i]);
                tmp[i] = C[i] * (tmp[i] - 1) * y[i];
            }
            XTv(tmp, g);
            for (int i = 0; i < w_size; i++) g[i] = w[i] + g[i];
            if (!regularize_bias) g[w_size - 1] -= w[w_size - 1];
        }

        @Override
        public void Hv(double[] s, double[] Hs) {
            int w_size = n;
            Arrays.fill(Hs, 0, w_size, 0);
            // Blocks of eight rows: their xTs (which only depend on s), then their updates of Hs
            // in row order, while the rows are still in cache.
            double[] coef = hvCoef;
            int i = 0;
            for (; i + 7 < l; i += 8) {
                dot8(s, i, coef);
                for (int k = i; k < i + 8; k++) coef[k] = C[k] * D[k] * coef[k];
                axpy8(coef, i, Hs);
            }
            for (; i < l; i++) {
                double xTs = dot(s, x[i]);
                xTs = C[i] * D[i] * xTs;
                axpy1(xTs, x[i], Hs);
            }
            for (i = 0; i < w_size; i++) Hs[i] = s[i] + Hs[i];
            if (!regularize_bias) Hs[w_size - 1] -= s[w_size - 1];
        }

        @Override
        public int get_nr_variable() {
            return n;
        }

        @Override
        public void get_diag_preconditioner(double[] M) {
            int w_size = n;
            for (int i = 0; i < w_size; i++) M[i] = 1;
            if (!regularize_bias) M[w_size - 1] = 0;
            for (int i = 0; i < l; i++) {
                double[] a = x[i];
                double ci = C[i], di = D[i];
                for (int j = 0; j < n; j++) M[j] += a[j] * a[j] * ci * di;
            }
        }

        @Override
        public double linesearch_and_update(double[] w, double[] s, MutableDouble f, double[] g, double alpha) {
            double sTs = 0;
            double wTs = 0;
            double gTs = 0;
            double eta = 0.01;
            int w_size = n;
            int max_num_linesearch = 20;
            double fold = f.get();
            Xv(s, tmp);
            for (int i = 0; i < w_size; i++) {
                sTs += s[i] * s[i];
                wTs += s[i] * w[i];
                gTs += s[i] * g[i];
            }
            if (!regularize_bias) {
                sTs -= s[w_size - 1] * s[w_size - 1];
                wTs -= s[w_size - 1] * w[w_size - 1];
            }
            int num_linesearch;
            for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
                double loss = 0;
                for (int i = 0; i < l; i++) {
                    double inner_product = tmp[i] * alpha + wx[i];
                    loss += C_times_loss(i, inner_product);
                }
                f.set(loss + (alpha * alpha * sTs + wTw) / 2.0 + alpha * wTs);
                if (f.get() - fold <= eta * alpha * gTs) {
                    for (int i = 0; i < l; i++) wx[i] += alpha * tmp[i];
                    break;
                } else {
                    alpha *= 0.5;
                }
            }
            if (num_linesearch >= max_num_linesearch) {
                f.set(fold);
                return 0;
            } else {
                for (int i = 0; i < w_size; i++) w[i] += alpha * s[i];
            }
            wTw += alpha * alpha * sTs + 2 * alpha * wTs;
            return alpha;
        }
    }
}
