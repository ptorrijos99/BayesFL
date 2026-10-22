package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * The averaged network of Scutari and Nagarajan (2013), the bnlearn baseline for
 * combining many learned structures: keep every pair whose edge strength (the
 * fraction of inputs adjacent on it) exceeds a threshold estimated from the
 * strengths themselves, and orient it the way most inputs do.
 *
 * The threshold fits the empirical CDF F of the strengths of all pairs with the
 * CDF of an ideal two-point distribution, mass 1-t at 0 and t at 1, in L1 over
 * [0,1). The best constant 1-t is the median of F(x) for x uniform on [0,1), and
 * the threshold is the strength at which F reaches it. With r inputs the
 * strengths are multiples of 1/r, so F is constant on [k/r,(k+1)/r) and the
 * median is taken over F(0), F(1/r), ..., F((r-1)/r).
 *
 * With a fixed threshold of 1/2 this is the bootstrap majority of Friedman et
 * al. (1999), i.e. the majority skeleton. Pairs are added by decreasing
 * strength, in the majority direction, reversed when that direction would close
 * a cycle (the reverse never can, since the graph was acyclic before).
 */
public class AveragedNetwork {

    /** Threshold chosen by the last build(). */
    public double threshold;

    public Dag build(PairWeights w) {
        int n = w.n(), r = w.r();
        long total = (long) n * (n - 1) / 2;
        long[] atMost = new long[r + 1];            // pairs with c_skel = k
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) atMost[w.countSkel(i, j)]++;
        for (int k = 1; k <= r; k++) atMost[k] += atMost[k - 1];

        double[] f = new double[r];                 // F(k/r), k = 0..r-1
        for (int k = 0; k < r; k++) f[k] = atMost[k] / (double) total;
        double[] s = f.clone();
        Arrays.sort(s);
        double c = r % 2 == 1 ? s[r / 2] : 0.5 * (s[r / 2 - 1] + s[r / 2]);
        int kStar = 0;
        while (kStar < r && f[kStar] < c - 1e-12) kStar++;
        threshold = kStar / (double) r;

        List<int[]> keep = new ArrayList<>();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (w.countSkel(i, j) > kStar) keep.add(new int[]{i, j});
        keep.sort((p, q) -> Integer.compare(w.countSkel(q[0], q[1]), w.countSkel(p[0], p[1])));

        List<Node> v = w.nodes();
        Dag out = new Dag(v);
        for (int[] p : keep) {
            int a = p[0], b = p[1];
            if (w.countDir(b, a) > w.countDir(a, b)) { int t = a; a = b; b = t; }
            Node x = v.get(a), y = v.get(b);
            if (out.paths().existsDirectedPath(y, x)) { Node t = x; x = y; y = t; }
            out.addEdge(Edges.directedEdge(x, y));
        }
        return out;
    }
}
