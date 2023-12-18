package org.albacete.simd.mctsbn;

import org.albacete.simd.utils.Problem;

import java.util.*;

public class TreeNode implements Comparable<TreeNode> {

    public final Integer node;
    private TreeNode parent;
    private final Map<TreeNode,TreeNode> children = new HashMap<>();
    private final Set<Integer> childrenIDs = new HashSet<>();

    private final LinkedHashSet<Integer> order;
    private final Problem problem;

    private int numVisits = 0;
    private double totalReward = 0;

    private double UCTSCore = 0;
    private double exploitationScore;
    private double explorationScore;

    private boolean fullyExpanded;
    private boolean isExpanded = false;

    MCTSBN mctsbn;

    /**
     * Constructor for the rest of the nodes.
     *
     * @param node The index of the variable that the node represents.
     * @param parent The parent node.
     */
    public TreeNode(Integer node, TreeNode parent) {
        this.node = node;
        this.parent = parent;
        this.order = new LinkedHashSet<>(parent.order);
        this.order.add(node);

        this.problem = parent.problem;
        this.mctsbn = parent.mctsbn;
        this.parent.addChild(this);

        this.fullyExpanded = isTerminal();
    }

    /**
     * Constructor for the root node.
     *
     * @param problem The Problem object with the information of the problem.
     * @param mctsbn The MCTSBN algorithm.
     * @return The root node.
     */
    public TreeNode(Problem problem, MCTSBN mctsbn) {
        this.node = -1;
        this.parent = null;
        this.order = new LinkedHashSet<>();

        this.problem = problem;
        this.mctsbn = mctsbn;

        this.fullyExpanded = true;
    }

    public TreeNode getParent() {
        return parent;
    }

    public Map<TreeNode,TreeNode> getChildren() {
        return children;
    }

    public Set<Integer> getChildrenIDs() {
        return childrenIDs;
    }

    public List<Integer> getPossibleChilds(ArrayList<Integer> orderPGES){
        List<Integer> possibleActions = new ArrayList<>();
        for(Integer var : orderPGES){
            if(!order.contains(var)){
                possibleActions.add(var);
            }
        }
        return possibleActions;
    }

    public void addChild(TreeNode child){
        // If the child is not in the tree, add it to the tree and back propagate the reward and the visits to the parent
        if (!this.childrenIDs.contains(child.node)) {
            this.children.put(child,child);
            this.childrenIDs.add(child.node);
            this.isExpanded = true;
            child.setParent(this);
        }
        // If the child is in the tree, fuse the trees
        else {
            // Fusion of the trees
            TreeNode childInTree = this.children.get(child);
            childInTree.addRewardAndVisits(child.totalReward, child.numVisits);

            for (TreeNode childChild : child.children.keySet()) {
                childInTree.addChild(childChild);
            }
        }
    }

    public void updateUCT() {
        if(this.parent == null && this.fullyExpanded){
            UCTSCore = Double.NEGATIVE_INFINITY;
        } else {
            exploitationScore = mctsbn.EXPLOITATION_CONSTANT * (this.getTotalReward() / this.getNumVisits());
            explorationScore = mctsbn.EXPLORATION_CONSTANT * Math.sqrt(Math.log(this.parent.getNumVisits()) / this.getNumVisits());

            UCTSCore = exploitationScore + explorationScore;
        }
    }

    public void backPropagate(double reward){
        this.totalReward += reward;
        this.numVisits += 1;

        if (this.parent != null) {
            this.parent.backPropagate(reward);
        }

        updateUCT();
    }

    public void addRewardAndVisits(double reward, int numVisits) {
        this.totalReward += reward;
        this.numVisits += numVisits;
        updateUCT();
    }

    public void setTotalReward(double totalReward) {
        this.totalReward = totalReward;
        updateUCT();
    }

    public int getNumVisits() {
        return numVisits;
    }

    public double getTotalReward() {
        return totalReward;
    }

    public boolean isFullyExpanded() {
        return fullyExpanded;
    }

    public void setFullyExpanded(boolean fullyExpanded) {
        this.fullyExpanded = fullyExpanded;
    }

    public LinkedHashSet<Integer> getOrder() {
        return order;
    }

    public boolean isExpanded() {
        return isExpanded;
    }

    public boolean isTerminal(){
        return order.size() == problem.getVariables().size();
    }

    public void setParent(TreeNode parent) {
        this.parent = parent;
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder(50);
        print(buffer, "", "");
        return buffer.toString();
    }

    private void print(StringBuilder buffer, String prefix, String childrenPrefix) {
        buffer.append(prefix);
        buffer.append("N" + node);

        String results;
        if(this.parent == null){
            results = "  \t" + this.getNumVisits() + "   BDeu " + exploitationScore + ", totalReward " + this.getTotalReward() + ", numVisits " + this.getNumVisits();
        } else {
            results = "  \t" + this.getNumVisits() + "   UCT " + UCTSCore + ",   BDeu " + exploitationScore + ",   Exploration " + explorationScore + ", totalReward " + this.getTotalReward() + ", numVisits " + this.getNumVisits();
        }
        buffer.append(results);

        buffer.append('\n');
        for (Iterator<TreeNode> it = children.keySet().iterator(); it.hasNext(); ) {
            TreeNode next = it.next();
            if (it.hasNext()) {
                next.print(buffer, childrenPrefix + "├── ", childrenPrefix + "│   ");
            } else {
                next.print(buffer, childrenPrefix + "└── ", childrenPrefix + "    ");
            }
        }
    }

    @Override
    public int compareTo(TreeNode o) {
        return Double.compare(this.UCTSCore, o.UCTSCore);
    }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(!(obj instanceof TreeNode other))
            return false;

        return Objects.equals(this.node, other.node);
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 29 * hash + this.node;
        return hash;
    }
}