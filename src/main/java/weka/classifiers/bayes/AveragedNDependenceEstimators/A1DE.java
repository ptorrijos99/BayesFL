/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    AODE.java
 *    Copyright (C) 2003
 *    Algorithm developed by: Geoff Webb
 *    Code written by: Janice Boughton & Zhihai Wang & Nayyar Zaidi
 */

package weka.classifiers.bayes.AveragedNDependenceEstimators;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;
import java.util.Vector;


/**
<!-- globalinfo-start -->
 * AODE achieves highly accurate classification by averaging over all of a small space 
 * of alternative naive-Bayes-like models that have weaker (and hence less detrimental) 
 * independence assumptions than naive Bayes. The resulting algorithm is computationally 
 * efficient while delivering highly accurate classification on many learning  tasks. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/> G. Webb, J. Boughton, Z. Wang (2005). 
 * Not So Naive Bayes: Aggregating One-Dependence Estimators. Machine Learning. 58(1):5-24.<br/>
 * <br/>
 * Further papers are available at<br/> http://www.csse.monash.edu.au/~webb/.<br/>
 * <br/>
 * Default frequency limit set to 1.
 * <p/>
<!-- globalinfo-end -->
 * 
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Webb2005,
 *    author = {G. Webb and J. Boughton and Z. Wang},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {5-24},
 *    title = {Not So Naive Bayes: Aggregating One-Dependence Estimators},
 *    volume = {58},
 *    year = {2005}
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 *
<!-- options-start -->
 * Valid options are: <p/>
 * 
 *  <pre> -F &lt;int&gt;
 *  Impose a frequency limit for superParents (default is 1). 
 *  </pre>
 * 
 *  <pre> -M &lt;int&gt;
 *  Specify a weight to use with m-estimate (default is 1). 
 *  </pre>
 *  
 *  <pre>
 *  Use A1DEUpdateable classifier for incremental learning so that probabilities are 
 *  calculated at classification time that will speed up learning but slows 
 *  down classification. 
 *  Numeric attributes are not supported in A1DEUpdateable version.
 *  </pre>
 *  
 *  <pre> -S (Optional) &lt;int&gt;
 *  Specify critical value of specialization-generalization for 
 *  Subsumption Resolution (default is false).
 *  Results in lowering bias and increasing variance of classification. 
 *  Recommended for large training data.
 *  See: 
 *  &#64;inproceedings{Zheng2006,
 *    author = {Fei Zheng and Geoffrey I. Webb},
 *    booktitle = {Proceedings of the Twenty-third International Conference on Machine  Learning (ICML 2006)},
 *    pages = {1113-1120},
 *    publisher = {ACM Press},
 *    title = {Efficient Lazy Elimination for Averaged-One Dependence Estimators},
 *    year = {2006},
 *    ISBN = {1-59593-383-2}
 * }
 *  </pre>
 * 
 *  <pre> -W (Optional) &lt;int&gt;
 *  Weighted AODE. Uses mutual information between attribute and the class as weight of
 *  each SPODE (default is false).
 *  Results in lowering bias and increasing variance of classification.
 *  Recommended for large training data.
 *  Can not use weighting for A1DEUpdateable classifier.
 *  See:  
 *  &#64;inproceedings{Jiang2006,
 *    author = {L. Jiang and H. Zhang},
 *    booktitle = {Proceedings of the 9th Biennial Pacific Rim International Conference on Artificial Intelligence, PRICAI 2006},
 *    pages = {970-974},
 *    series = {LNAI},
 *    title = {Weightily Averaged One-Dependence Estimators},
 *    volume = {4099},
 *    year = {2006}
 * }
 *  </pre>
 *  
<!-- options-end -->
 *
 * @author Janice Boughton (jrbought@csse.monash.edu.au)
 * @author Zhihai Wang (zhw@csse.monash.edu.au)
 * @author Nayyar Zaidi (nayyar.zaidi@monash.edu)
 * @version $Revision: 2 $
 */

public class A1DE extends AbstractClassifier implements 
OptionHandler, WeightedInstancesHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 9197439980415113523L;

	/** The discretization filter  */
	protected weka.filters.supervised.attribute.Discretize m_Disc = null;

	/**
	 * The frequency of two attribute-values occurring together for each class.
	 * Only unique combinations are stored.
	 */
	private double[] m_2vCondiCounts;

	/**
	 * The frequency of two attribute-values occurring together
	 */
	private double[] m_2vCondiCountsNoClass;

	/**
	 * The frequency of two attribute-values occuring together for each class.
	 * Stores reversed probabilities of m_2vCondiCounts, i.e., if 
	 * m_2vCondiCounts store P(x1|x2,y), m_2vCondiCountsOpp stores P(x2|x1,y).
	 * Used only in non-incremental version  
	 */
	private double[] m_2vCondiCountsOpp;

	/** The frequency of each attribute value occuring for each class */
	private double[] m_1vCondiCounts;

	/** 
	 * The probability of each attribute value conditioned on each class. 
	 * Used only in non-incremental version
	 */
	private double[] m_1vCondiCountsNB;

	/**
	 * Offsets to index combinations with 2 values (the ones in m_2vCondiCounts).
	 * An attribute-value is chosen for this level of offsets if it's the largest
	 * from a combination of two. E.g.: m_2vOffsets[7] + 2;
	 */
	private int[] m_2vOffsets;

	/** The number of times each class value occurs in the dataset */
	private double[] m_ClassCounts;

	/** The m-Estimate of the probabilities of each class.
	 * Used only in non-incremental version 
	 */
	private double[] m_ClassProbabilities;

	/** The sums of attribute-class counts, if there are no missing values for att, then
	 *  m_SumForCounts[classVal][att] will be the same as m_ClassCounts[classVal] 
	 */
	private double[][] m_SumForCounts;

	/** The number of classes */
	private int m_NumClasses;

	/** The number of attributes in dataset, including class */
	private int m_NumAttributes;

	/** The number of instances in the dataset */
	private int m_NumInstances;

	/** The index of the class attribute */
	private int m_ClassIndex;

	/** The dataset */
	public Instances m_Instances;

	/**
	 * The total number of values (including an extra for each attribute's 
	 * missing value, which are included in m_CondiCounts) for all attributes 
	 * (not including class). E.g., for three atts each with two possible values,
	 * m_TotalAttValues would be 9 (6 values + 3 missing).
	 * This variable is used when allocating space for m_CondiCounts matrix.
	 */
	private int m_TotalAttValues;

	/** The starting index (in the m_CondiCounts matrix) of the values for each
	 * attribute */
	private int[] m_StartAttIndex;

	/** The number of values for each attribute */
	private int[] m_NumAttValues;

	/** The frequency of each attribute value for the dataset */
	private double[] m_Frequencies;

	/** The number of valid class values observed in dataset 
	 *  -- with no missing classes, this number is the same as m_NumInstances.
	 */
	private double m_SumInstances;

	/** (Input paramters) An att's frequency must be this value or more to be a superParent */
	private int m_Limit = 1;

	/** (Input paramters) value for m in m-estimate */
	private double m_Weight = 1;	

	/** Initialize SPODE probabilities to some value to avoid underflow or overflow */
	private double probInitializerAODE = 1;

	/** Initialize SPODE probabilities to some value to avoid underflow or overflow */
	private double probInitializer = 1;

	/** (Input paramters) Do Subsumption Resolution */
	private boolean m_SubsumptionResolution = false;

	/** the critical value for the specialization-generalization */
	private int m_Critical = 100;	

	/** (Input paramters) Do Weighted AODE */
	private boolean m_WeightedAODE = false;

	/** The array of mutual information between each attribute and class */
	private double[] m_mutualInformation;

	/** Calculate conditional probability at training time or testing time. */
	protected static boolean m_Incremental = false;

	/** Use Discretization. */
	protected static boolean m_UseDiscretization = true;

	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "AODE achieves highly accurate classification by averaging over "
				+ "all of a small space of alternative naive-Bayes-like models that have "
				+ "weaker (and hence less detrimental) independence assumptions than "
				+ "naive Bayes. The resulting algorithm is computationally efficient "
				+ "while delivering highly accurate classification on many learning  "
				+ "tasks.\n\n"
				+ "For more information, see\n\n" + getTechnicalInformation().toString() 
				+ "\n\n"
				+ "Further papers are available at\n"
				+"  http://www.csse.monash.edu.au/~webb/.\n\n"
				+ "Use m-estimate for smoothing base probability estimates with" 
				+ "a default of 1 (m value can changed via option -M).\n  "
				+ "Default mode is non-incremental that is probabilites are computed "
				+ "at learning time. An incremental version can be used via option -I. \n"
				+ "Default frequency limit set to 1. \n"
				+ "Subsumption Resolution can be achieved by using -S option. \n"
				+ "Weighting of SPODE can be done by using -W option. Weights are calculated "
				+ "based on mutual information between attribute and the class. The weighting "
				+ "scheme is developed by L. Jiang and H. Zhang\n";
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "G. Webb and J. Boughton and Z. Wang");
		result.setValue(Field.YEAR, "2005");
		result.setValue(Field.TITLE, "Not So Naive Bayes: Aggregating One-Dependence Estimators");
		result.setValue(Field.JOURNAL, "Machine Learning");
		result.setValue(Field.VOLUME, "58");
		result.setValue(Field.NUMBER, "1");
		result.setValue(Field.PAGES, "5-24");

		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data
	 * @throws Exception if the classifier has not been generated
	 * successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		m_Instances = new Instances(instances);
		m_Instances.deleteWithMissingClass();

		// Discretize instances if required
		if (m_UseDiscretization) {
			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setInputFormat(m_Instances);
			m_Instances = weka.filters.Filter.useFilter(m_Instances, m_Disc);
		}
 
		// reset variable for this fold
		m_SumInstances = 0;
		m_ClassIndex = instances.classIndex();
		m_NumInstances = m_Instances.numInstances();
		m_NumAttributes = m_Instances.numAttributes();
		m_NumClasses = m_Instances.numClasses();

		// allocate space for attribute reference arrays
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes];

		m_TotalAttValues = 0;
		for(int i = 0; i < m_NumAttributes; i++) {
			if(i != m_ClassIndex) {
				m_StartAttIndex[i] = m_TotalAttValues;
				m_NumAttValues[i] = m_Instances.attribute(i).numValues();
				m_TotalAttValues += m_NumAttValues[i] + 1;
				// + 1 so room for missing value count
			} else {
				// m_StartAttIndex[i] = -1;  // class isn't included
				m_NumAttValues[i] = m_NumClasses;
			}
		}

		// allocate space for counts and frequencies
		// fill in the offsets that we need to index attribute-value combinations
		m_2vOffsets = new int[m_TotalAttValues];
		int nextIndex = 0, curAtt = 0, toAdd = 0;
		for (int i = 0; i < m_NumAttributes; i++) {
			if (i != m_ClassIndex) {
				for (int j = 0; j < m_NumAttValues[i] + 1; j++){ // each attribute-value
					// and missing value
					m_2vOffsets[curAtt] = nextIndex;
					// work out the offset for the *next* attribute-value
					nextIndex = m_2vOffsets[curAtt] + toAdd;
					curAtt++;
				}
			}
			toAdd += m_NumAttValues[i] + 1; // +1 for missing attVal
		}
		// If an offset starts at n, then there are n-1 locations before it
		// Since we work out the offset of the attribute value following the
		// current one, after the last loop nextIndex = number of two-valued
		// combinations

		/* Allocate space for counts C(.,.) and if non-incremental Classifier 
		 * is checked compute probabilities P(.,.)		 
		 */

		/*
		 *    m_2vCondiCounts in Incremental Version: 
		 * ---------- ---------- ---------- ---------- ----------    ----------
		 *| C(x2,x1) | C(x3,x2) | C(x3,x1) | C(x4,x3) | C(x4,x2) |..| C(x5,x1) |
		 * ---------- ---------- ---------- ---------- ----------    ----------
		 *
		 *    m_1vCondiCounts in Incremental Version: 
		 *     ------- ------- ------- ------- -------
		 *    | C(x1) | C(x2) | C(x3) | C(x4) | C(x5) |
		 *     ------- ------- ------- ------- -------
		 */
		m_2vCondiCounts = new double[nextIndex * m_NumClasses];
		m_1vCondiCounts = new double[m_TotalAttValues * m_NumClasses];

		// Additional data structure used in the program
		m_ClassCounts = new double[m_NumClasses];		
		m_SumForCounts = new double[m_NumClasses][m_NumAttributes];
		m_Frequencies = new double[m_TotalAttValues];

		if (m_SubsumptionResolution) {
			/*
			 *  Sumbsumption Resolution Flag is set. 
			 *  Allocate space for storing Count(x1,x2)
			 */
			m_2vCondiCountsNoClass = new double[nextIndex];
		}

		// calculate the counts
		for(int k = 0; k < m_NumInstances; k++) {
			addToCounts(m_Instances.instance(k));
		}

		// allocate memory for mutual information between attribute and class
		m_mutualInformation = new double[m_NumAttributes];
		for (int i = 0; i < m_NumAttributes; i++) {
			m_mutualInformation[i] = 1;
		}

		if (m_WeightedAODE && !m_Incremental) {
			/*
			 * Weighted AODE Flag is set. 
			 * Compute mutual information between each attribute and class 
			 */
			boolean nonZeroFlag = true;
			for (int att = 0; att < m_NumAttributes; att++) {
				if (att == m_ClassIndex) {
					m_mutualInformation[att] = 0;
					continue;
				}
				m_mutualInformation[att] = mutualInfo(att);

				if (m_mutualInformation[att] != 0) {
					nonZeroFlag = false;
				}
			}

			if (nonZeroFlag) {
				for (int att = 0; att < m_NumAttributes; att++) {
					if (att == m_ClassIndex)
						continue;
					m_mutualInformation[att] = (double)1/m_NumClasses;
				}
			}

			Utils.normalize(m_mutualInformation);	
		}

		// initialize probabilities
		probInitializer = Double.MAX_VALUE;
		probInitializerAODE = Double.MAX_VALUE/m_NumAttributes;

		// calculate conditional probability
		if (!m_Incremental) {
			/*
			 *    Following probabilities are also conditioned on y
			 *    
			 *    m_2vCondiCounts in Incremental Version: 
			 * ---------- ---------- ---------- ---------- ----------    ----------
			 *| P(x2|x1) | P(x3|x2) | P(x3|x1) | P(x4|x3) | P(x4|x2) |..| P(x5|x1) |
			 * ---------- ---------- ---------- ---------- ----------    ----------
			 *     
			 *    m_2vCondiCountsOpp in Incremental Version: 
			 * ---------- ---------- ---------- ---------- ----------    ----------
			 *| P(x1|x2) | P(x2|x3) | P(x1|x3) | P(x3|x4) | P(x2|x4) |..| P(x1|x5) |
			 * ---------- ---------- ---------- ---------- ----------    ----------
			 *
			 *    m_1vCondiCounts in Incremental Version: 
			 *     --------- --------- --------- --------- ---------
			 *    | P(x1,y) | P(x2,y) | P(x3,y) | P(x4,y) | P(x5,y) |
			 *     --------- --------- --------- --------- ---------
			 *     
			 *    m_1vCondiCountsNB in Incremental Version: 
			 *     --------- --------- --------- --------- ---------
			 *    | P(x1|y) | P(x2|y) | P(x3|y) | P(x4|y) | P(x5|y) |
			 *     --------- --------- --------- --------- ---------
			 *          
			 */			
			m_2vCondiCountsOpp = new double[nextIndex * m_NumClasses];			
			m_1vCondiCountsNB = new double[m_TotalAttValues * m_NumClasses];

			m_ClassProbabilities = new double[m_NumClasses];
			calcConditionalProbs();
		}

		// free up some space
		m_Instances.delete();

	}

	/** 
	 * This function converts the counts in m_CondiCounts to conditional 
	 * probability estimates. This method is called during model building, so
	 * the conditional probabilities don't have to be calculated at the test
	 * time.
	 * @throws Exception 
	 */
	private void calcConditionalProbs() throws Exception {

		/* Local variables */
		double pCount = 0, cCount = 0, pcCount = 0;
		double missingForAtt1 = 0, missingForAtt2 = 0;
		double conditionalProb = 0, oppositeCondProb = 0;
		double jointProb = 0, condProb = 0;

		/* 
		 * Calculate conditional probabilities of the form P(xp|xc,y) or
		 * P(xc|xp,y) from the counts
		 */		
		for (int parent = m_NumAttributes - 1; parent >= 0; parent--) {	
			if(parent == m_ClassIndex) continue;			
			for (int pval = m_StartAttIndex[parent]; 
					pval < (m_StartAttIndex[parent] + m_NumAttValues[parent]); pval++) {

				for (int child = parent - 1; child >= 0; child--) {	
					if(child == m_ClassIndex) continue;					
					for (int cval = m_StartAttIndex[child]; 
							cval < (m_StartAttIndex[child] + m_NumAttValues[child]); cval++) {

						for (int classval = 0; classval < m_NumClasses; classval++) {

							pcCount = getCountFromTable(classval, pval, cval);

							pCount =  getCountFromTable(classval, pval, pval);
							cCount =  getCountFromTable(classval, cval, cval);

							/*
							 *               C(xc,xp,y) + m/Card(xp)
							 * P(xp|xc,y) = -------------------------
							 *               (C(xc,y) - ?(xp)) + m
							 * 
							 * where Card(xp) is cardinality of xp and 
							 *       ?(xp) are the number of missing values of xp
							 * 
							 * Store P(xp|xc) in m_2vCondiCounts
							 */
							missingForAtt2 = m_2vCondiCounts[((m_2vOffsets[m_StartAttIndex[parent] 
									+ m_NumAttValues[parent]] + cval) * m_NumClasses) + classval];
							conditionalProb = (pcCount + m_Weight/m_NumAttValues[parent]) / 
									((cCount - missingForAtt2) + m_Weight);
							m_2vCondiCounts[(m_2vOffsets[pval] + cval) * 
							                m_NumClasses + classval] = conditionalProb;	

							/*
							 *               C(xc,xp,y) + m/Card(xc)
							 * P(xc|xp,y) = -------------------------
							 *               (C(xp,y) - ?(xc)) + m
							 * 
							 * where Card(xc) is cardinality of xc and 
							 *       ?(xc) are the number of missing values of xc
							 *       
							 * Store P(xc|xp) in m_2vCondiCountsOpp
							 */
							missingForAtt1 = m_2vCondiCounts[((m_2vOffsets[pval] + 
									m_StartAttIndex[child] + m_NumAttValues[child]) * 
									m_NumClasses) + classval];
							oppositeCondProb = (pcCount + m_Weight/m_NumAttValues[child]) / 
									((pCount - missingForAtt1) + m_Weight);
							m_2vCondiCountsOpp[(m_2vOffsets[pval] + cval) * 
							                   m_NumClasses + classval] = oppositeCondProb;						
						} // ends classval

					} // ends cval				
				} // ends child

			} // ends pval
		} // ends parent       	

		/*
		 * Compute joint probabilities of the form P(x1,y) and conditional 
		 * probabilities of the form P(x1|y)
		 */	
		for (int Att1 = 0; Att1 < m_NumAttributes; Att1++) {
			if (Att1 == m_ClassIndex) continue;

			double missing4ParentAtt = 0;
			missing4ParentAtt = m_Frequencies[m_StartAttIndex[Att1] + m_NumAttValues[Att1]];

			for(int a1val = m_StartAttIndex[Att1]; 
					a1val < (m_StartAttIndex[Att1] + m_NumAttValues[Att1]); a1val++) {

				for (int classval = 0; classval < m_NumClasses; classval++) {		
					pCount = getCountFromTable(classval, a1val, a1val);	

					/*
					 *                 C(x1,x1,y) + m/(Card(x1)*Card(y))
					 * 		P(x1,y) = ------------------------------------
					 *                      (N - ?(x1,y)) + m
					 * 
					 *    Store result in m_1vCondiCounts		  
					 */
					jointProb = ((pCount + m_Weight/(m_NumClasses * m_NumAttValues[Att1])) / 
							((m_SumInstances - missing4ParentAtt) + m_Weight));
					m_1vCondiCounts[(a1val * m_NumClasses) + classval] = jointProb;

					/*
					 *                  C(x1,x1,y) + m/|x1|
					 * 		P(x1|y) = -----------------------
					 *                   (#N - ?x1)  + m 
					 * 
					 *    Store result in m_1vCondiCountsNB
					 */
					condProb = (pCount + m_Weight/m_NumAttValues[Att1]) / 
							(m_SumForCounts[classval][Att1] + m_Weight);
					m_1vCondiCountsNB[(a1val * m_NumClasses) + classval] = condProb;
				}
			}
		}

		/*
		 * Convert class counts into probabilities that is compute P(y)
		 */
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			m_ClassProbabilities[classVal] = MEsti(m_ClassCounts[classVal], 
					m_SumInstances, m_NumClasses);
		}

	}

	/**
	 * Get count from m_1vCondiCount or m_2vCondiCount.
	 * 
	 * @param the class value
	 * @param Index of the parent
	 * @param index of the child
	 * @return count stored in m_1vCondiCount or m_2vCondiCount
	 */
	public double getCountFromTable(int classVal, int pIndex, int childIndex) {
		if (pIndex == childIndex)  
			return m_1vCondiCounts[(pIndex * m_NumClasses) + classVal];
		else
			return m_2vCondiCounts[((m_2vOffsets[pIndex] + childIndex) * m_NumClasses) + classVal];
	}

	/**
	 * Updates the classifier with the given instance.
	 *
	 * @param instance the new training instance to include in the model 
	 */
	public void updateClassifier(Instance instance) {

		if (m_UseDiscretization) {
			m_Disc.input(instance);
			instance = m_Disc.output();
		}

		if (m_Incremental)
			this.addToCounts(instance);
		else {
			System.err.println("Classifier is not incremental");			
		}
	}

	/** 
	 * Puts an instance's values into m_CondiCounts, m_ClassCounts and 
	 * m_SumInstances.
	 *
	 * @param instance  the instance whose values are to be put into the counts
	 *                  variables
	 */
	private void addToCounts(Instance instance) {

		if(instance.classIsMissing())
			return;   // ignore instances with missing class

		int classVal = (int)instance.classValue();
		double weight = (double)instance.weight();

		m_ClassCounts[classVal] += weight;
		m_SumInstances += weight;

		// store instance's att val indexes in an array, b/c accessing it 
		// in loop(s) is more efficient
		int [] attIndex = new int[m_NumAttributes];
		for(int i = 0; i < m_NumAttributes; i++) {
			if(i == m_ClassIndex)
				attIndex[i] = -1;  // we don't use the class attribute in counts
			else {
				if(instance.isMissing(i))
					attIndex[i] = m_StartAttIndex[i] + m_NumAttValues[i];
				else {
					attIndex[i] = m_StartAttIndex[i] + (int)instance.value(i);
					m_SumForCounts[classVal][i] += weight;
				}
			}
		}

		for(int Att1 = m_NumAttributes - 1; Att1 >= 0; Att1--) {
			int Att1Index = attIndex[Att1]; 
			if(Att1Index == -1)
				continue;   // avoid pointless looping as Att1 is currently the class attribute

			m_Frequencies[Att1Index] += weight;
			m_1vCondiCounts[(Att1Index * m_NumClasses) + classVal] += weight;

			// no need to grab this again inside the loop
			int Att1ValOffset = m_2vOffsets[Att1Index];

			for(int Att2 = Att1 - 1; Att2 >= 0 ; Att2--) {
				int Att2Index = attIndex[Att2];
				if(attIndex[Att2] != -1) {
					int endIndex = (Att1ValOffset + Att2Index) * m_NumClasses;
					m_2vCondiCounts[endIndex + classVal] += weight;

					if (m_SubsumptionResolution) {
						m_2vCondiCountsNoClass[Att1ValOffset + Att2Index] += weight;
					}
				}
			}
		}

	}

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if there is a problem generating the prediction
	 */
	@Override
	public double [] distributionForInstance(Instance instance) throws Exception {

		if (m_UseDiscretization) {
			m_Disc.input(instance);
			instance = m_Disc.output();
		}

		// accumulates posterior probabilities for each class
		double[] probs = new double[m_NumClasses];

		double[][] spodeProbs = new double[m_NumAttributes][m_NumClasses];
		double[][] classParentsFreq = new double[m_NumAttributes][m_NumClasses];
		double parentFreq;
		int parentCount = 0;
		int pIndex, childIndex, comboIndex;

		int[] SpecialGeneralArray = new int[m_NumAttributes];
		double counts;

		/* 
		 * Store instance's att indexes in an int array, 
		 * so accessing them is more efficient in loop(s). 
		 */
		int [] attIndex = new int[m_NumAttributes];
		for(int att = 0; att < m_NumAttributes; att++) {
			if(instance.isMissing(att) || att == m_ClassIndex)
				attIndex[att] = -1;   // can't use class or missing values in calculations
			else
				attIndex[att] = m_StartAttIndex[att] + (int)instance.value(att);
		}

		// -1 indicates attribute is not a generalization of any other attributes
		for (int i = 0; i < m_NumAttributes; i++) {
			SpecialGeneralArray[i] = -1;
		}

		/* Do subsumption Resolution */
		if (m_SubsumptionResolution) {			

			// calculate the specialization-generalization array
			for (int i = 0; i < m_NumAttributes; i++) {
				// skip i if it's the class or is missing
				if (attIndex[i] == -1)  continue;

				for (int j = 0; j < m_NumAttributes; j++) {
					// skip j if it's the class, missing, is i or a generalization of i
					if ((attIndex[j] == -1) || (i == j) || (SpecialGeneralArray[j] == i))
						continue;

					if (i < j) {
						counts = m_2vCondiCountsNoClass[m_2vOffsets[attIndex[j]] + attIndex[i]]; 
					} else {
						counts = m_2vCondiCountsNoClass[m_2vOffsets[attIndex[i]] + attIndex[j]];
					}

					// check j's frequency is above critical value
					if (m_Frequencies[attIndex[j]] > m_Critical) {

						// skip j if the frequency of i and j together is not equivalent
						// to the frequency of j alone
						if (m_Frequencies[attIndex[j]] == counts) {

							// if attributes i and j are both a specialization of each other
							// avoid deleting both by skipping j
							if ((m_Frequencies[attIndex[j]] == m_Frequencies[attIndex[i]]) && (i < j)) {
								continue;
							} else {
								// set the specialization relationship
								SpecialGeneralArray[i] = j;
								break; // break out of j loop because a specialization has been found
							}
						}
					}

				}
			}

		}

		/* Get the prior probabilities first, each attribute has a turn of being the parent. */
		for (int parent = 0; parent < m_NumAttributes; parent++) {
			pIndex = attIndex[parent];
			if (pIndex == -1)
				continue;

			// delete the generalization attributes.
			if (SpecialGeneralArray[parent] != -1) 
				continue;

			int local1vOffset = pIndex * m_NumClasses;

			// check that the att value has a frequency of m_Limit or greater
			if (m_Frequencies[pIndex] >= m_Limit) {
				parentCount++;
				// find the number of missing values for parent's attribute
				double missing4ParentAtt = m_Frequencies[m_StartAttIndex[parent] + 
				                                         m_NumAttValues[parent]];				

				// calculate the prior probability -- P(parent & classVal)
				for (int classVal = 0; classVal < m_NumClasses; classVal++) {
					if (m_Incremental) {
						/* 
						 * Get the count C(x1,x1,y) from m_CondiCount and compute the joint
						 * probability as
						 * 
						 *             C(x1,x1,y) + m/(Card(x1)*Card(y))
						 * P(x1,y) = ------------------------------------
						 *                   (N - ?(x1,y)) + m
						 *   
						 */	
						parentFreq = m_1vCondiCounts[local1vOffset + classVal];						
						spodeProbs[parent][classVal] = probInitializerAODE * 
								m_mutualInformation[parent] *
								MEsti(parentFreq, (m_SumInstances - missing4ParentAtt), 
										(m_NumClasses * m_NumAttValues[parent]));						
						classParentsFreq[parent][classVal] = parentFreq;						
					} else {
						/* No need to compute probabilities from the counts.
						 * The probability P(x1,x1,y) is stored in m_1vCondiCounts */
						spodeProbs[parent][classVal] = probInitializerAODE * 
								m_mutualInformation[parent] * m_1vCondiCounts[local1vOffset + classVal];
					}	
				}	
			} else {
				for (int classVal = 0; classVal < m_NumClasses; classVal++) {
					parentFreq = m_1vCondiCounts[local1vOffset + classVal];
					classParentsFreq[parent][classVal] = parentFreq;
				}
			}
		}

		// check that at least one parent was used, else do NB
		if (parentCount < 1) {
			return NBconditionalProb(instance, attIndex);
		}

		for (int parent = 1; parent < m_NumAttributes; parent++) {
			pIndex = attIndex[parent];
			if (pIndex == -1)
				continue;

			// delete the generalization attributes.
			if (SpecialGeneralArray[parent] != -1) 
				continue;

			int pOffset = m_2vOffsets[pIndex];

			for (int child = 0; child < parent; child++) {
				childIndex = attIndex[child];
				if (childIndex == -1)
					continue;

				// delete the generalization attributes.
				if (SpecialGeneralArray[child] != -1) 
					continue;

				comboIndex = (pOffset + childIndex) * m_NumClasses;
				double missingForParent = 0, missingForChild = 0, countForAttOpp = 0, countForAtt = 0;

				for (int classVal = 0; classVal < m_NumClasses; classVal++) {			

					if (m_Incremental) {
						// Get the count C(x1,x2) from m_2vCondiCounts array
						countForAtt = m_2vCondiCounts[comboIndex + classVal];

						/* 
						 * Compute probability
						 * 
						 *              C(xp,xc) + m/|xp|
						 * P(xp|xc) = ---------------------
						 *              {#(xc) - ?xp} + m
						 */
						missingForParent = m_2vCondiCounts[((m_2vOffsets[m_StartAttIndex[parent] + 
						                                                 m_NumAttValues[parent]] + 
						                                                 childIndex) * m_NumClasses) + 
						                                                 classVal];
						spodeProbs[child][classVal] *= (countForAtt + m_Weight/m_NumAttValues[parent]) / 
								((classParentsFreq[child][classVal] - missingForParent) + m_Weight);	

						/* 
						 * Compute probability
						 *  
						 *              C(xp,xc) + m/|xc|
						 *  P(xc|xp) = --------------------
						 *              {C(xp) - ?xc} + m
						 */
						missingForChild = m_2vCondiCounts[((m_2vOffsets[pIndex] + m_StartAttIndex[child] + 
								m_NumAttValues[child]) * m_NumClasses) + classVal];
						spodeProbs[parent][classVal] *= (countForAtt + m_Weight/m_NumAttValues[child]) / 
								((classParentsFreq[parent][classVal] - missingForChild) + m_Weight);

					} else {
						/* 
						 * No need to compute probabilities from counts.
						 * The probability P(xc|xp) is stored in m_2vCondiCountsOpp, and
						 * the probability P(xp|xc) is stored in m_2vCondiCounts
						 */
						countForAtt = m_2vCondiCounts[comboIndex + classVal];						
						spodeProbs[child][classVal] *= countForAtt;

						countForAttOpp = m_2vCondiCountsOpp[comboIndex + classVal];
						spodeProbs[parent][classVal] *= countForAttOpp;
					}
				}
			}
		}

		/* add all the probabilities for each class */
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			for (int i = 0; i < m_NumAttributes; i++) {
				probs[classVal] += spodeProbs[i][classVal] + Double.MIN_VALUE;
			}			
		}

		Utils.normalize(probs);

		return probs;
	}

	/**
	 * Calculates the probability of the specified class for the given test
	 * instance, using naive Bayes.
	 *
	 * @param instance the instance to be classified
	 * @param classVal the class for which to calculate the probability
	 * @return predicted class probability
	 * @throws Exception 
	 */
	public double[] NBconditionalProb(Instance instance, int[] attIndex) throws Exception {

		double[] probs = new double[m_NumClasses];
		double countForAtt;

		// calculate the prior probability
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			if (m_Incremental) {
				probs[classVal] = probInitializer * 
						MEsti(m_ClassCounts[classVal], m_SumInstances, m_NumClasses);
			} else {
				// No need to compute the probability
				probs[classVal] = probInitializer * m_ClassProbabilities[classVal];								
			}
		}

		// consider effect of each att value
		for (int child = 0; child < m_NumAttributes; child++) {
			// determine correct index for att in m_CondiCounts
			int childIndex = attIndex[child];
			if (attIndex[child] == -1)
				continue;

			for (int classVal = 0; classVal < m_NumClasses; classVal++) {
				if (m_Incremental) {
					/* Get the count C(x1,x1,y) from m_CondiCount and compute the conditional
					 * probability as
					 * 
					 *                  C(x1,x1,y) + m/|x1|
					 * 		P(x1|y) = -----------------------
					 *                  (#N - ?x1)  + m 
					 */
					countForAtt = m_1vCondiCounts[(childIndex * m_NumClasses) + classVal];
					probs[classVal] *= MEsti(countForAtt, m_SumForCounts[classVal][child], 
							m_NumAttValues[child]);					
				} else {
					/* No need to compute conditional probability from the counts, 
					 * This is stored at m_1fCondiCountsNB, so just retrieve the value
					 */
					probs[classVal] *= m_1vCondiCountsNB[(childIndex * m_NumClasses) + classVal];
				}
			}
		}

		Utils.normalize(probs);
		return probs;  
	}

	/**
	 * Computes mutual information between each attribute and class attribute.
	 *
	 * @param att is the attribute
	 * @return the conditional mutual information between son and parent given class
	 * @throws Exception 
	 */
	private double mutualInfo(int att) throws Exception {

		double mutualInfo = 0;
		double[] PriorsClass = new double[m_NumClasses];
		double[] PriorsAttribute = new double[m_NumAttValues[att]];
		double[][] PriorsClassAttribute = new double[m_NumClasses][m_NumAttValues[att]];

		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			PriorsClass[classVal] = MEsti(m_ClassCounts[classVal], 
					m_SumInstances, m_NumClasses);			
		}

		for (int i = 0; i < m_NumAttValues[att]; i++) {
			PriorsAttribute[i] = MEsti(m_Frequencies[m_StartAttIndex[att] + i], 
					m_SumInstances, m_NumAttValues[att]);			
		}

		/* 
		 * Get the count C(x1,x1,y) from m_CondiCount and compute the joint
		 * probability as
		 * 
		 *             C(x1,x1,y) + m/(Card(x1)*Card(y))
		 * P(x1,y) = ------------------------------------
		 *                   (N - ?(x1,y)) + m
		 *   
		 */	
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			for (int i = 0; i < m_NumAttValues[att]; i++) {

				int local1vOffset = (m_StartAttIndex[att] + i) * m_NumClasses;
				double missing4ParentAtt = m_Frequencies[m_StartAttIndex[att] + m_NumAttValues[att]];	

				PriorsClassAttribute[classVal][i] = MEsti(m_1vCondiCounts[local1vOffset + classVal], (m_SumInstances - missing4ParentAtt), (m_NumClasses * m_NumAttValues[att]));				
			}
		}

		for (int i = 0; i < m_NumClasses; i++) {
			for (int j = 0; j < m_NumAttValues[att]; j++) {
				mutualInfo += PriorsClassAttribute[i][j] * log2(PriorsClassAttribute[i][j], PriorsClass[i] * PriorsAttribute[j]);
			}
		}

		return mutualInfo;
	}

	/**
	 * Performs m-estimation
	 */
	public double MEsti(double freq1, double freq2, double numValues) throws Exception {
		double mEsti = (freq1 + m_Weight / numValues) / (freq2 + m_Weight);
		return mEsti;
	}

	/**
	 * compute the logarithm whose base is 2.
	 *
	 * @param x numerator of the fraction.
	 * @param y denominator of the fraction.
	 * @return the natual logarithm of this fraction.
	 */
	private double log2(double x, double y){

		if (x < Utils.SMALL || y < Utils.SMALL)
			return 0.0;
		else
			return Math.log(x/y)/Math.log(2);
	}

	/**
	 * Returns an enumeration describing the available options
	 *
	 * @return an enumeration of all the available options
	 */
	@Override
	public Enumeration listOptions() {

		Vector newVector = new Vector(4);

		newVector.addElement(new Option("\tOutput debugging information\n", "D", 0,"-D"));
		newVector.addElement(new Option("\tImpose a frequency limit for superParents \t (default is 1)", "F", 1,"-F <int>"));
		newVector.addElement( new Option("\tSpecify a weight to use with m-estimate (default is 1) \n", "M", 1, "-M <double>"));
		newVector.addElement( new Option("\tSpecify a critical value for specialization-generalilzation SR (default is 100) \n", "S", 1, "-S <int>"));
		newVector.addElement( new Option("\tSpecify if to use weighted AODE \n", "W", 0, "-W"));		

		return newVector.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 * 
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -F &lt;int&gt;
	 *  Impose a frequency limit for superParents
	 *  (default is 1)</pre>
	 * 
	 * <pre> -M &lt;int&gt;
	 *  Specify a weight to use with m-estimate
	 *  (default is 1)</pre> 
	 *  
	 * <pre> -S (Optional) &lt;int&gt;
	 *  Specify critical value of specialization-generaliztion for 
	 *  subsumption resolution
	 *  (default is 100)
	 *  Results in lowering bias and increasing variance
	 *  
	 * <pre> -W (Optional) &lt;int&gt;
	 *  Do Weighted AODE
	 *  Results in lowering bias and increasing variance
	 *  
	 * 
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		String Freq = Utils.getOption('F', options);
		if (Freq.length() != 0) 
			m_Limit = Integer.parseInt(Freq);
		else
			m_Limit = 1;

		String Weight = Utils.getOption('M', options);
		if (Weight.length() != 0)
			m_Weight = Double.parseDouble(Weight);			   
		else 
			m_Weight = 1;

		String m_SROption = Utils.getOption('S', options);		
		if (m_SROption.length() != 0) {
			m_SubsumptionResolution = true;
			m_Critical = Integer.parseInt(m_SROption);					
		} else { 
			m_Critical = 100;
		}

		m_WeightedAODE = Utils.getFlag('W', options);		

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String [] getOptions() {
		Vector result  = new Vector();

		result.add("-F");
		result.add("" + m_Limit);

		result.add("-M");
		result.add("" + m_Weight);

		if (m_SubsumptionResolution)    
			result.add("-S");

		if (m_WeightedAODE)    
			result.add("-W");

		return (String[]) result.toArray(new String[result.size()]);
	}

	/**
	 * Sets the frequency limit
	 *
	 * @param f the frequency limit
	 */
	public void setFrequencyLimit(int f) {
		m_Limit = f;
	}

	/**
	 * Gets the frequency limit.
	 *
	 * @return the frequency limit
	 */
	public int getFrequencyLimit() {
		return m_Limit;
	}

	/**
	 * Sets the weight for m-estimate
	 *
	 * @param w the weight
	 */
	public void setWeight(double weight) {
		if (weight > 0)
			m_Weight = weight;
		else
			System.out.println("Weight must be greater than 0!");
	}

	/**
	 * Gets the weight used in m-estimate
	 *
	 * @return the frequency limit
	 */
	public double getWeight() {
		return m_Weight;
	}

	/**
	 * Sets the Subsumption Resolution flag
	 *
	 * @param S the Subsumption Resolution flag
	 */
	public void setSubsumptionResolution(boolean S) {
		m_SubsumptionResolution = S;
	}

	/**
	 * Gets the Subsumption Resolution flag.
	 *
	 * @return the SubsumptionResolution flag
	 */
	public boolean getSubsumptionResolution() {
		return m_SubsumptionResolution;
	}

	/**
	 * Sets the Weighted AODE flag
	 *
	 * @param f the Weighted AODE flag
	 */
	public void setWeightedAODE(boolean W) {
		m_WeightedAODE = W;
	}

	/**
	 * Gets the Weighted AODE flag.
	 *
	 * @return the Weighted AODE flag
	 */
	public boolean getWeightedAODE() {
		return m_WeightedAODE;
	}

	/**
	 * Returns a description of the classifier.
	 *
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {

		StringBuffer text = new StringBuffer();
		text.append("The A1DE Classifier\n");
		if (m_Instances == null) {
			text.append(": No model built yet.");
		} else {
			try {
				for (int i = 0; i < m_NumClasses; i++)
					text.append("\nClass " + m_Instances.classAttribute().value(i) + 
							": Prior probability = " + 
							Utils.doubleToString(((m_ClassCounts[i] + 1) / 
									(m_SumInstances + m_NumClasses)), 4, 2));

				text.append("\n\nDataset: " + m_Instances.relationName() + "\n" + 
						"Instances: " + m_NumInstances + "\n" + "Attributes: " + m_NumAttributes + 
						"\n" + "Frequency limit for superParents: (F = " + m_Limit + ") \n");
				text.append("Correction: ");
				text.append("m-estimate (m = " + m_Weight + ")\n");
				text.append("Incremental Classifier Flag: (" + m_Incremental + ")\n");
				text.append("Subsumption Resolution Flag: (" + m_SubsumptionResolution + ")\n");
				text.append("Critical Value for Subsumption Resolution (" + m_Critical + ")\n");				
				text.append("Weighted AODE Flag: (" + m_WeightedAODE + ")\n");
			} catch (Exception ex) {
				text.append(ex.getMessage());
			}
		}

		return text.toString();
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 5516 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param args the options
	 */
	public static void main(String [] args) {
		runClassifier(new A1DE(), args);
	}	

}
