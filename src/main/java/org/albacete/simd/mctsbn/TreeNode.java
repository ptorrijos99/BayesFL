package org.albacete.simd.mctsbn;

import org.albacete.simd.utils.ProblemMCTS;

import java.util.*;

public class TreeNode implements Comparable<TreeNode> {

    public final Integer node;
    private final TreeNode parent;
    private final Set<TreeNode> children = new HashSet<>();
    private final Set<Integer> childrenIDs = new HashSet<>();

    private final LinkedHashSet<Integer> order;
    private final ProblemMCTS problem;
    private double localScore;

    private int numVisits = 0;
    private double totalReward = 0;
    private double UCTSCore = 0;

    private boolean fullyExpanded;
    private boolean isExpanded = false;

    MCTSBN mctsbn;

    public TreeNode(Integer node, TreeNode parent) {
        this.node = node;
        this.parent = parent;
        this.order = new LinkedHashSet<>(parent.order);
        this.order.add(node);

        this.problem = parent.problem;
        this.parent.addChild(this);
        
        this.numVisits = 0;
        this.totalReward = 0;
        this.fullyExpanded = isTerminal();

        /*
        if (node != -1) {
            // Evaluating the new node if its not the root
            HashSet<Integer> candidates = new HashSet<>(allVars.size());
            for (Integer candidate : allVars) {
                if (!order.contains(candidate)) {
                    candidates.add(candidate);
                }
            }
            this.localScore += hc.evaluate(this.node, candidates).bdeu;
        }
         */
    }

    public TreeNode(ProblemMCTS problem) {
        this.problem = problem;

        this.node = -1;
        this.parent = null;
        this.order = new LinkedHashSet<>();
        this.order.add(this.node);

        this.numVisits = 0;
        this.totalReward = 0;
        this.fullyExpanded = true;
    }

    public TreeNode getParent() {
        return parent;
    }

    public Set<TreeNode> getChildren() {
        return children;
    }
          
    public Set<Integer> getChildrenIDs() {
        return childrenIDs;
    }

    public TreeNode newChild(Integer node){
        return new TreeNode(node, this);
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
        this.children.add(child);
        this.childrenIDs.add(child.node);
        this.isExpanded = true;
    }

    public int getNumVisits() {
        return numVisits;
    }

    public double getTotalReward() {
        return totalReward;
    }

    public double getUCTScore() {
        return UCTSCore;
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

    public void addReward(double reward){
        totalReward += reward;
    }

    public void setTotalReward(double totalReward) {
        this.totalReward = totalReward;
    }

    public void incrementOneVisit(){
        this.numVisits++;
    }

    public void decrementOneVisit(){
        this.numVisits--;
    }

    public boolean isExpanded() {
        return isExpanded;
    }

    public boolean isTerminal(){
        return order.size() == problem.getVariables().size();
    }

    public void updateUCT() {
        if(this.parent == null && this.fullyExpanded){
            UCTSCore = Double.NEGATIVE_INFINITY;
        } else {
            double exploitationScore = mctsbn.EXPLOITATION_CONSTANT * (this.getTotalReward() / this.getNumVisits());
            double explorationScore = mctsbn.EXPLORATION_CONSTANT * Math.sqrt(Math.log(this.parent.getNumVisits()) / this.getNumVisits());

            UCTSCore = exploitationScore + explorationScore;
        }
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

        double exploitationScore = mctsbn.EXPLOITATION_CONSTANT * (this.getTotalReward() / this.getNumVisits());

        if(this.parent == null){
            results = "  \t" + this.getNumVisits() + "   BDeu " + exploitationScore;
        } else {
            double explorationScore = mctsbn.EXPLORATION_CONSTANT * Math.sqrt(Math.log(this.parent.getNumVisits()) / this.getNumVisits());
            results = "  \t" + this.getNumVisits() + "   UCT " + UCTSCore + ",   BDeu " + exploitationScore + ",   EXP " + explorationScore;
        }
        buffer.append(results);

        buffer.append('\n');
        for (Iterator<TreeNode> it = children.iterator(); it.hasNext();) {
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
