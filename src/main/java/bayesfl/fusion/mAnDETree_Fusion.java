package bayesfl.fusion;

import bayesfl.algorithms.mAnDETree_mAnDE;
import bayesfl.model.Model;
import bayesfl.model.mAnDETree;
import org.albacete.simd.mAnDE.mSPnDE;

import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

public class mAnDETree_Fusion implements Fusion {

    @Override
    public Model fusion(Model model1, Model model2) {
        return fusion (new Model[]{model1, model2});
    }

    @Override
    public Model fusion(Model[] models) {
        for (Model model : models) {
            if (!(model instanceof mAnDETree)) {
                throw new IllegalArgumentException("The models must be objects of the mAnDETree class to use mAnDETree_Fusion");
            }
        }

        ConcurrentHashMap<Object, mSPnDE> fusedSPnDEs = new ConcurrentHashMap<>();

        int i=0;
        for (Model model : models) {
            ConcurrentHashMap<Object, mSPnDE> mSPnDEs = ((mAnDETree) model).getModel();
            System.out.println("mSPnDEs " + i + ": " + mSPnDEs);
            
            for (Object key : mSPnDEs.keySet()) {
                // If the fused model contains the key, we add the children of the model to the fused model
                if (fusedSPnDEs.containsKey(key)) {
                    mSPnDE mSPnDE_fusion = fusedSPnDEs.get(key);
                    mSPnDE mSPnDE_model = mSPnDEs.get(key);

                    mSPnDE_fusion.moreChildren(mSPnDE_model.getChildren());
                }
                // If the fused model does not contain the key, we add the model to the fused model
                else {
                    fusedSPnDEs.put(key, mSPnDEs.get(key).copyDeep());
                }
            }
            i++;
        }

        System.out.println("\nmSPnDEs fusion: " + fusedSPnDEs);


        return new mAnDETree(fusedSPnDEs, ((mAnDETree) models[0]).getAlgorithm());
    }
}
