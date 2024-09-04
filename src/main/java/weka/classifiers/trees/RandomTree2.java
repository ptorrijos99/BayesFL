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
 *    RandomTree2.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package weka.classifiers.trees;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.concurrent.ConcurrentHashMap;
import org.albacete.simd.mAnDE.NodeInt;
import org.albacete.simd.mAnDE.mSP1DE;
import org.albacete.simd.mAnDE.mSPnDE;


public class RandomTree2 extends RandomTree {

    /**
     * The size of each bag sample, as a percentage of the training size
     *
     * Changued from int to double
     */
    protected double m_BagSizePercentDouble = 100;


    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercentDouble the bag size, as a percentage.
     */
    public void setBagSizePercentDouble(double newBagSizePercentDouble) {

        m_BagSizePercentDouble = newBagSizePercentDouble;
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public double getBagSizePercentDouble() {

        return m_BagSizePercentDouble;
    }

    public void toSP1DE(ConcurrentHashMap<Object, mSPnDE> mSPnDEs) {
        // If the tree is empty, return
        if (m_Tree.m_Attribute == -1) {
            return;
        }

        // Trees to be explored
        LinkedList<Tree> tbExplored = new LinkedList<>();
        tbExplored.add(m_Tree);
        mSPnDEs.putIfAbsent(m_Tree.m_Attribute, new mSP1DE(m_Tree.m_Attribute));

        while (!tbExplored.isEmpty()) {
            Tree node = tbExplored.poll();

            int id = node.m_Attribute;
            mSPnDE mSPnDE = mSPnDEs.get(id);

            // Add childs to the mSPnDE
            for (Tree m_Successor : node.m_Successors) {
                int child_id = m_Successor.m_Attribute;

                // If is not a leaf
                if (child_id != -1) {

                    // Add child
                    mSPnDE.moreChildren(child_id);

                    // Create child if neccesary and add parent as child
                    mSPnDEs.putIfAbsent(child_id, new mSP1DE(child_id));
                    mSPnDEs.get(child_id).moreChildren(id);

                    // Add node to tbExplored
                    tbExplored.add(m_Successor);
                }
            }
        }
    }

    /**
     *  Create the representation of the graph
     */
    public HashMap<Integer, NodeInt> toSP2DE() {
        HashMap<Integer, NodeInt> nodes = new HashMap<>();

        // Trees to be explored
        LinkedList<Tree> tbExplored = new LinkedList<>();
        tbExplored.add(m_Tree);

        while (!tbExplored.isEmpty()) {
            Tree node = tbExplored.poll();
            int id1 = node.m_Attribute;

            if (!nodes.containsKey(id1)) {
                nodes.put(id1, new NodeInt(id1));
            }

            // Add childs
            for (Tree m_Successor : node.m_Successors) {
                int id2 = m_Successor.m_Attribute;

                // If is not a leaf
                if (id2 != -1) {
                    // Add child
                    if (nodes.get(id2) == null) {
                        nodes.put(id2, new NodeInt(id2, nodes.get(id1)));
                    }
                    nodes.get(id1).addChild(nodes.get(id2));
                }
            }
        }

        return nodes;
    }
}
