package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * Per-triple weights for the immorality part of the distance between
 * equivalence classes.
 *
 * Two DAGs are Markov equivalent exactly when they share skeleton and
 * immoralities. So the natural distance between classes is the skeleton
 * distance plus |I(R) ^ I(G)|, with I the set of immoralities a->c<-b (a, b
 * non-adjacent). The skeleton part decomposes over pairs (PairWeights). The
 * immorality part decomposes over triples by the same identity:
 *
 *     (1/r) SUM_i |I(R) ^ I(G_i)| = (1/r) [ SUM_i |I(G_i)| + SUM_{t in I(R)} (r - 2 c_t) ]
 *
 * with c_t the number of inputs holding immorality t. Only triples that occur in
 * some input have c_t > 0; every other triple costs r, so they are not stored.
 *
 * Triples are keyed as (min(a,b), c, max(a,b)) over PairWeights indices.
 */
public class ImmoralityWeights {

    private final PairWeights w;
    private final int r;
    private final Map<Long, Integer> count = new HashMap<>();
    private long constant = 0;

    public ImmoralityWeights(List<Dag> inputs, PairWeights w) {
        this.w = w;
        this.r = inputs.size();
        for (Dag g : inputs) {
            for (Node c : g.getNodes()) {
                List<Node> pa = g.getParents(c);
                for (int i = 0; i < pa.size(); i++)
                    for (int j = i + 1; j < pa.size(); j++) {
                        if (g.isAdjacentTo(pa.get(i), pa.get(j))) continue;
                        count.merge(key(w.idx(pa.get(i)), w.idx(c), w.idx(pa.get(j))), 1, Integer::sum);
                        constant++;
                    }
            }
        }
    }

    long key(int a, int c, int b) {
        int lo = Math.min(a, b), hi = Math.max(a, b);
        long n = w.n();
        return ((long) lo * n + c) * n + hi;
    }

    /** Cost weight of immorality a->c<-b. Negative when a majority of inputs hold it. */
    public double weight(Node a, Node c, Node b) {
        Integer k = count.get(key(w.idx(a), w.idx(c), w.idx(b)));
        return r - 2.0 * (k == null ? 0 : k);
    }

    /** Mean immorality distance of a candidate to the inputs. */
    public double distance(Dag g) {
        double acc = constant;
        for (Node c : g.getNodes()) {
            List<Node> pa = g.getParents(c);
            for (int i = 0; i < pa.size(); i++)
                for (int j = i + 1; j < pa.size(); j++)
                    if (!g.isAdjacentTo(pa.get(i), pa.get(j))) acc += weight(pa.get(i), c, pa.get(j));
        }
        return acc / r;
    }

    /**
     * Change in the immorality objective from adding x->y to g (x, y not adjacent).
     * Created: x->y<-p for every other parent p of y not adjacent to x.
     * Destroyed: x->c<-y for every common child c, now shielded by the new edge.
     */
    public double gainAdd(Dag g, Node x, Node y) {
        double d = 0;
        for (Node p : g.getParents(y))
            if (!p.equals(x) && !g.isAdjacentTo(p, x)) d += weight(x, y, p);
        for (Node c : g.getChildren(x))
            if (g.isParentOf(y, c)) d -= weight(x, c, y);
        return d;
    }

    /** Change from removing x->y from g: the mirror image of gainAdd. */
    public double gainRemove(Dag g, Node x, Node y) {
        double d = 0;
        for (Node p : g.getParents(y))
            if (!p.equals(x) && !g.isAdjacentTo(p, x)) d -= weight(x, y, p);
        for (Node c : g.getChildren(x))
            if (g.isParentOf(y, c)) d += weight(x, c, y);
        return d;
    }
}
