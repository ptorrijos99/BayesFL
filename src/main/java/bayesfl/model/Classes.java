package bayesfl.model;

import bayesfl.data.Data;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A minimal model representing only the structure information needed
 * for federating AnDE-style synthetic class values.
 * <p>
 * This model contains:
 * - The list of attribute index combinations (for AnDE).
 * - The synthetic class mappings observed locally (per combination).
 * </p>
 * It is used solely for the purpose of discovering structure prior to
 * federated training. It contains no classifiers or parameter trees.
 */
public class Classes implements Model {

    /**
     * The list of attribute index combinations (e.g. all pairs for A1DE).
     */
    private final List<int[]> combinations;

    /**
     * The synthetic class maps observed locally, one per combination.
     */
    private final List<Map<String, Integer>> syntheticClassMaps;

    /**
     * Constructor.
     *
     * @param combinations List of attribute index combinations.
     * @param syntheticClassMaps Local class mappings per combination.
     */
    public Classes(List<int[]> combinations, List<Map<String, Integer>> syntheticClassMaps) {
        this.combinations = combinations;
        this.syntheticClassMaps = syntheticClassMaps;
    }

    /**
     * Gets the combinations used.
     *
     * @return The list of attribute index arrays.
     */
    public List<int[]> getCombinations() {
        return combinations;
    }

    /**
     * Gets the synthetic class mappings.
     *
     * @return The list of class maps.
     */
    public List<Map<String, Integer>> getSyntheticClassMaps() {
        return syntheticClassMaps;
    }

    /**
     * Gets the model. In this case, the model is a combination of
     * combinations and synthetic class maps.
     *
     * @return The model, which is an array of combinations and synthetic class maps.
     */
    @Override
    public Object getModel() {
        return this;
    }

    /**
     * This operation is not supported for this type of model.
     *
     * @param model Ignored.
     */
    @Override
    public void setModel(Object model) {
        throw new UnsupportedOperationException("Classes model does not support setModel.");
    }

    /**
     * No statistics are saved for structure-only models.
     *
     * @param operation Ignored.
     * @param epoch Ignored.
     * @param path Ignored.
     * @param nClients Ignored.
     * @param id Ignored.
     * @param data Ignored.
     * @param iteration Ignored.
     * @param time Ignored.
     */
    @Override
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        // No-op
    }

    /**
     * This model does not support scoring.
     *
     * @return Always throws.
     */
    @Override
    public double getScore() {
        throw new UnsupportedOperationException("Classes model does not support scoring.");
    }

    /**
     * This model does not support scoring.
     *
     * @param data Ignored.
     * @return Always throws.
     */
    @Override
    public double getScore(Data data) {
        throw new UnsupportedOperationException("Classes model does not support scoring.");
    }

    /**
     * Returns a string representation of the Classes model.
     *
     * @return A string representation of the model.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Classes Model:\n");
        for (int i = 0; i < combinations.size(); i++) {
            sb.append("Combination ").append(i).append(": ");
            sb.append("Attributes: ").append(Arrays.toString(combinations.get(i))).append(", ");
            sb.append("Synthetic Class Map: ").append(syntheticClassMaps.get(i)).append("\n");
        }
        return sb.toString();
    }
}
