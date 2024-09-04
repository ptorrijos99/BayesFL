/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */
/**
 *    MCT_Fusion.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 */

package bayesfl.fusion;

import bayesfl.model.BN;
import bayesfl.model.MCT;
import bayesfl.model.Model;
import org.albacete.simd.mctsbn.TreeNode;

import java.util.ArrayList;

public class MCT_Fusion implements Fusion {
    /**
     * Perform the fusion of two MCT models.
     *
     * @param model1 The first MCT model to fuse.
     * @param model2 The second MCT model to fuse.
     * @return The global MCT model fused.
     */
    @Override
    public Model fusion(Model model1, Model model2) {
        return fusion(new Model[]{model1,model2});
    }

    /**
     * Perform the fusion of many MCT models.
     *
     * @param models The array of MCT Model to fuse.
     * @return The global MCT model fused.
     */
    @Override
    public Model fusion(Model[] models) {
        for (Model model : models) {
            if (!(model instanceof MCT)) {
                throw new IllegalArgumentException("The models must be objects of the MCT class to use MCT_Fusion");
            }
        }

        ArrayList<BN> candidates = new ArrayList<>();
        for (Model model : models) {
            candidates.add(((MCT) model).getBestBN());
        }

        MCT model = (MCT) models[0];
        MCT fused = new MCT((TreeNode) model.getModel(), candidates);

        for (int i = 1; i < models.length; i++) {
            model = (MCT) models[i];

            TreeNode fusedNode = (TreeNode) fused.getModel();
            TreeNode modelNode = (TreeNode) model.getModel();

            for (TreeNode child : modelNode.getChildren().keySet()) {
                fusedNode.addChild(child);
            }
        }

        return fused;
    }
}


