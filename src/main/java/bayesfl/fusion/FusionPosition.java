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
 *    FusionPosition.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.fusion;

import bayesfl.model.Model;

public class FusionPosition implements Fusion {

    public int position = -1;

    public FusionPosition() {
    }

    public FusionPosition(int position) {
        this.position = position;
    }

    /**
     * Perform the fusion of two models.
     *
     * @param model1 The first model to fuse.
     * @param model2 The second model to fuse.
     * @return If position is 0 , return model1. If position is 1, -1 (default) or another number, return model2.
     */
    @Override
    public Model fusion(Model model1, Model model2) {
        if (position == 0) return model1;
        else return model2;
    }

    /**
     * Perform the fusion of many models.
     *
     * @param models The array of Model to fuse.
     * @return If position is between 0 and models.length-1, return the model in that position. If position is -1,
     * return the last model of the array. If position is greater than models.length-1, return the first model of
     * the array.
     */
    @Override
    public Model fusion(Model[] models) {
        if (position >=0 && position < models.length) return models[position];
        else if (position == -1) return models[models.length-1];
        else return models[0];
    }

    /**
     * Set the position of the model that will be returned.
     * @param position The position of the model that will be returned.
     */
    public void setPosition(int position) {
        this.position = position;
    }
}
