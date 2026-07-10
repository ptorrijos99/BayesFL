package bayesfl.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WDPTHeaderTest {
    @Test
    void headerContainsParamDpColumns() {
        String header = WDPT.HEADER; // promote the existing header string to a public constant
        for (String col : new String[]{"epsilonParam", "clipC", "sigma", "localSteps", "rounds", "epsilonTotal"}) {
            assertTrue(header.contains(col), "header must contain " + col);
        }
    }
}
