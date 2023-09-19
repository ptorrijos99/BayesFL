package consensusBN;

import java.util.HashSet;

import edu.cmu.tetrad.graph.Node;

public class SubSet extends HashSet<Node> {
	
	private static final long serialVersionUID = 4569314863278L;
	public static final int TEST_NOT_EVALUATED=0;
	public static final int TEST_TRUE=1;
	public static final int TEST_FALSE=-1;
	
	public int firstTest=TEST_NOT_EVALUATED;
	public int secondTest=TEST_NOT_EVALUATED;
	
	public SubSet() {
		super();
	}
	
	public SubSet(SubSet other) {
		super(other);
	}
}