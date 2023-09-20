package org.albacete.simd.mctsbn;

import org.albacete.simd.utils.Problem;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

public class State {

    private final Integer node;

    private final List<Integer> order;
    
    private final List<Integer> allVars;
    
    private double localScore;
    
    private final HillClimbingEvaluator hc;

    private final Problem problem;

    public State(Integer node, List<Integer> order, List<Integer> allVars, Problem problem, double localScore, HillClimbingEvaluator hc){
        this.node = node;
        this.order = order;
        this.allVars = allVars;
        this.problem = problem;
        this.localScore = localScore;
        this.hc = hc;
        
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
    }

    public List<Integer> getPossibleActions(){
        List<Integer> possibleActions = new ArrayList<>();
        for(Integer var : allVars){
            if(!order.contains(var)){
                possibleActions.add(var);
            }
        }
        return possibleActions;
    }
    
    public List<Integer> getPossibleActionsbyOrder(ArrayList<Integer> orderPGES){
        List<Integer> possibleActions = new ArrayList<>();
        for(Integer var : orderPGES){
            if(!order.contains(var)){
                possibleActions.add(var);
            }
        }
        return possibleActions;
    }

    public State takeAction(Integer action){
        List<Integer> newOrder = new ArrayList<>(order);
        newOrder.add(action);

        return new State(action, newOrder, allVars, problem, localScore, hc);
    }
    
    public void setLocalScore(double localScore) {
        this.localScore = localScore;
    }

    public boolean isTerminal(){
        return order.size() == problem.getVariables().size();
    }
    
    public double getLocalScore(){ 
        return localScore;
    }

    public Integer getNode(){
        return this.node;
    }

    public List<Integer> getOrder() {
        return order;
    }
    
    public HillClimbingEvaluator getHC(){
        return hc;
    }

    @Override
    public String toString() {
     return "State with node: " + node + "----- and order: " + order;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        State state = (State) o;
        return Objects.equals(node, state.node) && Objects.equals(order, state.order);
    }

    @Override
    public int hashCode() {
        return Objects.hash(node, order);
    }
}
