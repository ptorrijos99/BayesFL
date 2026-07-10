package bayesfl.privacy;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ZCdpGaussianCalibrationTest {

    private static final double DELTA = 1e-5;

    @Test
    void rhoEpsilonRoundTrip() {
        for (double eps : new double[]{0.5, 1.0, 2.0, 5.0}) {
            double rho = ZCdpGaussianCalibration.rhoForEpsilon(eps, DELTA);
            double back = ZCdpGaussianCalibration.epsilonForRho(rho, DELTA);
            assertEquals(eps, back, 1e-9, "round trip at eps=" + eps);
            assertTrue(rho > 0, "rho positive at eps=" + eps);
        }
    }

    @Test
    void noiseMultiplierRealisesTargetEpsilon() {
        long T = 200;
        double eps = 1.0;
        double sigma = ZCdpGaussianCalibration.noiseMultiplier(eps, DELTA, T);
        double rho = ZCdpGaussianCalibration.rhoSpent(T, sigma);
        double realised = ZCdpGaussianCalibration.epsilonForRho(rho, DELTA);
        assertEquals(eps, realised, 1e-9);
    }

    @Test
    void moreStepsMeansMoreNoise() {
        double a = ZCdpGaussianCalibration.noiseMultiplier(1.0, DELTA, 100);
        double b = ZCdpGaussianCalibration.noiseMultiplier(1.0, DELTA, 400);
        assertTrue(b > a, "sigma grows with T at fixed epsilon");
    }

    @Test
    void smallerEpsilonMeansMoreNoise() {
        double loose = ZCdpGaussianCalibration.noiseMultiplier(5.0, DELTA, 100);
        double tight = ZCdpGaussianCalibration.noiseMultiplier(0.5, DELTA, 100);
        assertTrue(tight > loose, "sigma grows as epsilon shrinks");
    }
}
