package bayesfl.fusion;

import bayesfl.model.Classes;
import bayesfl.model.Model;

import java.util.*;

/**
 * Federates synthetic class mappings (AnDE joint labels)
 * from all clients into a global, consistent mapping.
 * <p>
 * This fusion ensures that all clients will use the same
 * class map per combination for training, with consistent
 * label ordering and cardinalities.
 * </p>
 */
public class Classes_Fusion implements Fusion {

    /**
     * Fuses two models (unused, uses full array version).
     */
    @Override
    public Model fusion(Model model1, Model model2) {
        return fusion(new Model[]{model1, model2});
    }

    /**
     * Fuses all local `Classes` models into a global one.
     *
     * @param models Array of local models.
     * @return A single `Classes` model with federated combinations and synthetic class maps.
     */
    @Override
    public Model fusion(Model[] models) {
        if (models == null || models.length == 0) {
            throw new IllegalArgumentException("Empty model list.");
        }

        // Extract number of combinations
        int nCombinations = ((Classes) models[0]).getCombinations().size();

        // Initialize sets to collect all unique labels per combination
        List<Set<String>> globalLabelSets = new ArrayList<>();
        for (int i = 0; i < nCombinations; i++) {
            globalLabelSets.add(new LinkedHashSet<>());
        }

        // Aggregate all synthetic labels observed locally
        for (Model model : models) {
            Classes local = (Classes) model;
            List<Map<String, Integer>> localMaps = local.getSyntheticClassMaps();

            for (int i = 0; i < nCombinations; i++) {
                globalLabelSets.get(i).addAll(localMaps.get(i).keySet());
            }
        }

        // Convert each label set into a consistent class map
        List<Map<String, Integer>> globalClassMaps = new ArrayList<>();
        for (Set<String> labelSet : globalLabelSets) {
            // Sort labels to ensure consistent ordering
            List<String> sortedLabels = new ArrayList<>(labelSet);
            Collections.sort(sortedLabels);

            Map<String, Integer> map = new LinkedHashMap<>();
            int index = 0;
            for (String label : sortedLabels) {
                map.put(label, index++);
            }
            globalClassMaps.add(map);
        }

        // Reuse combinations from any model (they are assumed identical)
        List<int[]> globalCombinations = ((Classes) models[0]).getCombinations();

        return new Classes(globalCombinations, globalClassMaps);
    }
}
