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
 *    Node.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class NodeInt {

    /**
     * ID of the Node.
     */
    private int id;

    /**
     * Name of the parent of the node.
     */
    private NodeInt parent;

    /**
     * HashMap with the children of the node.
     */
    private HashMap<Integer, NodeInt> children;

    /**
     * Constructor. Creates a parent node, passing it the ID and name.
     *
     * @param id Node ID
     */
    public NodeInt(int id) {
        this.parent = this;
        this.id = id;
        this.children = new HashMap<>();
    }

    /**
     * Constructor. Creates a child node, passing it the ID and parent node.
     *
     * @param id Node ID
     * @param parent Parent of the node
     */
    public NodeInt(int id, NodeInt parent) {
        this(id);
        this.parent = parent;
    }

    /**
     * Add a child.
     *
     * @param child Child to add
     */
    public void addChild(NodeInt child) {
        this.children.put(child.getId(), child);
    }

    /**
     * Returns the array of children.
     *
     * @return The array of children
     */
    protected ArrayList<Integer> getChildrenArray() {
        ArrayList<Integer> array = new ArrayList<>();
        children.values().forEach((child) -> {
            array.add(child.id);
        });
        return array;
    }

    /**
     * Returns the array of children that are not "no".
     *
     * @param no The child not to return.
     * @return The array of children
     */
    protected ArrayList<Integer> getChildrenArray(Integer no) {
        ArrayList<Integer> array = new ArrayList<>();
        children.values().forEach((child) -> {
            if (!(child.id == no)) {
                array.add(child.id);
            }
        });
        return array;
    }

    /**
     * @return the ID
     */
    public int getId() {
        return id;
    }

    /**
     * @param id the ID to set
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * @return the parent
     */
    public NodeInt getParent() {
        return parent;
    }

    /**
     * @param padre the parent to set
     */
    public void setParent(NodeInt padre) {
        this.parent = padre;
    }

    /**
     * @return the children
     */
    public HashMap<Integer, NodeInt> getChildren() {
        return children;
    }

    /**
     * @param children the children to set
     */
    public void setChildren(HashMap<Integer, NodeInt> children) {
        this.children = children;
    }

}
